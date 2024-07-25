import torch
import time
from queue import PriorityQueue as PQ
from collections import deque
import bisect
from bisect import bisect_left
import torch.nn.functional as F

def genPriorityQueue(intvmgr):
    pq = PQ()
    pairs = [(intvmgr.getIntervalStart(i), i, intvmgr.isSpillable(i)) for i in range(len(intvmgr.interv_info))]
    for p in pairs:
        pq.put(p)
    return pq

def spill(cfg, i, regnum):
    cfg[i] = regnum

def assign(cfg, i, idx, intrvmgr, phy_intrvs):
    cfg[i] = idx
    phy_intrvs[idx] = (i, intrvmgr.isSpillable(i), intrvmgr.getSegs(i))

def evict(idx, phy_intrvs):
    tup = phy_intrvs[idx]
    phy_intrvs[idx] = (None, None, None)
    return tup

def intersect(segs1, segs2, i, j):
    assert i!= j
    assert(segs1 != None)
    if segs2 == None:
        return False
    for seg1 in segs1:
        c1 = (seg1[0] + seg1[1] - 1) / 2
        l1 = (seg1[1] - seg1[0] - 1) / 2
        for seg2 in segs2:
            c2 = (seg2[0] + seg2[1] - 1) / 2
            l2 = (seg2[1] - seg2[0] - 1)  / 2
            if (abs(c2 - c1) <= (l2 + l1)):
                return True
    return False

def legalization(cfg, intvmgr, net = None):
    if net!= None:
        s = time.time()
        prev_obj = net.objEval(cfg)
    mis_match = 0
    regnum = intvmgr.getRegNum()
    pq = genPriorityQueue(intvmgr)
    cfg = torch.argmax(cfg, dim = 1)
    spilled_count1 = 0
    spilled_count2 = 0
    phy_intrvs = []
    for i in range(regnum):
        phy_intrvs.append((None, None, None))
    # Linear scan modified
    while (not pq.empty()):
        start, i, spillable = pq.get()
        segs = intvmgr.interv_info[i][2]
        idx = cfg[i].item()
        if idx == regnum:
            spill(cfg, i, regnum)
            continue
        prev_id, _, phy_segs = phy_intrvs[idx]
        conflict = intersect(segs, phy_segs, i, prev_id)
        # Assign directly
        if not conflict:
            if phy_segs == None or phy_segs[-1][1] <= start:
                assign(cfg, i, idx, intvmgr, phy_intrvs)
            else:
                tup = evict(idx, phy_intrvs)
                assign(cfg, i, idx, intvmgr, phy_intrvs)
                for seg in tup[2]:
                    if seg[0] > start:
                        pq.put((seg[0], tup[0], tup[1]))
                        break
            continue
        active = []; inactive = []; lateruse = []
        for j, tup in enumerate(phy_intrvs):
            if intvmgr.masks[i][j] != 1:
                continue
            prev_id, phy_spillable, phy_segs = tup
            if intersect(segs, phy_segs, i, prev_id):
                if phy_spillable == 1:
                    active.append(j)
            else:
                if phy_segs == None or phy_segs[-1][1] <= start:
                    inactive.append(j)
                else:
                    lateruse.append(j)
        mis_match += 1
        # Cannot get a reg for free
        if spillable == 1:
            spill(cfg, i, regnum)
            spilled_count1 += 1
            continue
        # For unspillable short intervals
        if len(inactive) > 0:
            assign(cfg, i, inactive[0], intvmgr, phy_intrvs)
        else:
            if idx in active:
                targ = idx
            else:
                targ = active[0]
            assert(len(active) > 0)
            tup = evict(targ, phy_intrvs)
            spill(cfg, tup[0], regnum)
            spilled_count2 += 1
            assign(cfg, i, targ, intvmgr, phy_intrvs)
    cfg = F.one_hot(cfg, num_classes = regnum + 1)
    if net!= None:
        cur_obj = net.objEval(cfg)
        print("=========== legalization ==========")
        print("Mismatch count : %.6f, spilled_count : %d, %d,  Time : %.4f"%(mis_match, spilled_count1, spilled_count2, time.time() - s))
        print("Before %.6f, After %.6f"%(prev_obj, cur_obj))
    return cfg

def legalization_v2(cfg, intvmgr, net = None):
    if net!= None:
        s = time.time()
        prev_obj = net.objEval(cfg)
    thres = 10
    mis_match = 0; spilled_count1 = 0; spilled_count2 = 0
    regnum = intvmgr.getRegNum()
    pq = genPriorityQueue(intvmgr)
    cfg = torch.argmax(cfg, dim = 1)
    phy_intrvs = []; realloc = []
    fails = []
    for i in range(regnum):
        phy_intrvs.append((None, None, None))
    while (not pq.empty()):
        start, i, spillable = pq.get()
        segs = intvmgr.interv_info[i][2]
        idx = cfg[i].item()
        if idx == regnum:
            assert spillable
            spill(cfg, i, regnum)
            continue
        prev_id, _, phy_segs = phy_intrvs[idx]
        conflict = intersect(segs, phy_segs, i, prev_id)
        # Assign directly
        if not conflict:
            if phy_segs == None or phy_segs[-1][1] <= start:
                assign(cfg, i, idx, intvmgr, phy_intrvs)
            else:
                tup = evict(idx, phy_intrvs)
                assign(cfg, i, idx, intvmgr, phy_intrvs)
                for seg in tup[2]:
                    if seg[0] > start:
                        pq.put((seg[0], tup[0], tup[1]))
                        break
            continue
        if spillable:
            prev_id = phy_intrvs[idx][0]
            size1 = intvmgr.sizes[i]
            size2 = intvmgr.sizes[prev_id]
            if size1 < thres:
                spill(cfg, i, regnum)
            elif size2 < thres:
                evict(idx, phy_intrvs)
                spill(cfg, prev_id, regnum)
                assign(cfg, i, idx, intvmgr, phy_intrvs)
                if intvmgr.getIntervInfo("spillable", prev_id) == 0:
                    realloc.append(prev_id)
            else:
                spill(cfg, i, regnum)
                fails.append((i, prev_id))
        else:
            realloc.append(i)
            spill(cfg, i, regnum)
    # Fix these conflicts
    ori_realloc = realloc.copy()
    realloc, cfg, phy_segs, starts = refillSpillable(realloc, cfg, intvmgr)
    cfg = evictAndRefill(realloc, cfg, intvmgr, phy_segs, starts)
    for tup in fails:
        print("-------")
        print(tup)
      # print(intvmgr.getSegs(tup[0]))
      # print(intvmgr.getSegs(tup[1]))
    cfg = F.one_hot(cfg, num_classes = regnum + 1)

    if net!= None:
        cur_obj = net.objEval(cfg)
        print("=========== legalization ==========")
        print("Before %.6f, After %.6f"%(prev_obj, cur_obj))
    return cfg

def evictAndRefill(realloc, cfg, intvmgr, phy_segs, starts):
    cands2realloc = {}
    for i in realloc:
        seg = intvmgr.getSegs(i)[0]
        find = False
        for k in range(intvmgr.getRegNum()):
            if intvmgr.masks[i, k] != 1:
                continue
            idx = bisect_left(starts[k], seg[0])
            # Contains this small interval
            candtup = phy_segs[k][idx - 1]
            if idx > 0 and candtup[1] >= seg[1]:
                if intvmgr.getIntervInfo("spillable", candtup[2]) == 0:
                    continue
                if candtup[2] not in cands2realloc:
                    cands2realloc[candtup[2]] = []
                cands2realloc[candtup[2]].append(i)
                find = True
        assert find
    # Realloc in groups
    realloc_segs = [(intvmgr.getSegs(x)[0], x) for x in realloc]
    realloc_segs_sorted = sorted(realloc_segs, key = lambda x : x[0][0])
    def genGroup(segs_sorted):
        if len(segs_sorted) == 0:
            return [], []
        res1 = [segs_sorted[0]]; res2 = []
        for i in range(1, len(segs_sorted)):
            prev_seg = res1[-1]
            cur_seg = segs_sorted[i]
            if cur_seg[0][0] < prev_seg[0][1]:
                res2.append(cur_seg)
            else:
                res1.append(cur_seg)
        return res1, res2
    cur_group, next_group = genGroup(realloc_segs_sorted)
    while len(cur_group) > 0:
        cur_group_set = set([x[1] for x in cur_group])
        while len(cur_group_set) > 0:
            best_score = 1e8; best_id = None; b_active = None
            for key in cands2realloc.keys():
                active = [x for x in cands2realloc[key] if x in cur_group_set]
                if len(active) == 0:
                    continue
                score = intvmgr.calcReloadAdv(key, cfg).item() / len(active)
                if score < best_score:
                    best_id = key
                    best_score = score
                    b_active = set(active)
            cands2realloc.pop(best_id)
            for key in cands2realloc.keys():
                cands2realloc[key] = [x for x in cands2realloc[key] if x not in b_active]
            cur_group_set = cur_group_set - b_active
            for i in b_active:
                cfg[i] = cfg[best_id]
            assert best_id != 5328
            cfg[best_id] = intvmgr.getRegNum()
        cur_group, next_group = genGroup(next_group)
    return cfg

def refillSpillable(realloc, cfg, intvmgr):
    phy_segs = []; starts = []
    for i in range(intvmgr.getRegNum()):
        phy_segs.append([[2 * intvmgr.vplen, 2 * intvmgr.vplen, None]])
    for i in range(cfg.shape[0]):
        idx = cfg[i].item()
        if idx == intvmgr.getRegNum():
            continue
        phy_segs[idx] += [[x[0], x[1], i] for x in intvmgr.getSegs(i)]
    for i in range(intvmgr.getRegNum()):
        phy_segs[i] = sorted(phy_segs[i], key = lambda x : x[0])
        starts.append([x[0] for x in phy_segs[i]])
    left = []
    for i in realloc:
        seg = intvmgr.getSegs(i)[0]
        refilled = False
        for k in range(intvmgr.getRegNum()):
            if intvmgr.masks[i, k] != 1:
                continue
            idx = bisect_left(starts[k], seg[0])
            if idx < len(starts[k]) and seg[1] > starts[k][idx]:
                continue
            if idx > 0 and phy_segs[k][idx - 1][1] > seg[0]:
                continue
            refilled = True
            phy_segs[k].insert(idx, [seg[0], seg[1], i])
            starts[k].insert(idx, seg[0])
            cfg[i] = k
            break
        if not refilled:
            left.append(i)
    return left, cfg, phy_segs, starts

def refill(cfg, intvmgr, net = None):
    if net!= None:
        s = time.time()
        prev_obj = net.objEval(cfg)
    refill_count = 0
    regnum = intvmgr.getRegNum()
    cfg = cfg.argmax(dim = 1)
    phy_intrvs = []
    starts = []
    for i in range(regnum):
        phy_intrvs.append([])
        starts.append([])
    to_spill = [x for x in range(cfg.shape[0]) if cfg[x] == regnum]
    for i, idx, in enumerate(cfg):
        if i in to_spill:
            continue
        phy_intrvs[idx] += intvmgr.interv_info[i][2]
    for i in range(regnum):
        phy_intrvs[i] = sorted(phy_intrvs[i], key = lambda x : x[0])
        starts[i] = [x[0] for x in phy_intrvs[i]]
    with_score = [(x, intvmgr.calcReloadAdv(x, cfg)) for x in to_spill]
    with_score = sorted(with_score, key=lambda x: -x[1])
    for i, score in with_score:
        segs = intvmgr.interv_info[i][2]
        allocated = False
        new_score = intvmgr.calcReloadAdv(i, cfg)
        if new_score < 0:
            continue
        for k in range(regnum):
            conflict = False
            if intvmgr.masks[i][k] != 1:
                continue
            for seg in segs:
                idx = bisect_left(starts[k], seg[0])
                if idx < len(starts[k]) and seg[1] > starts[k][idx]:
                    conflict = True
                    break
                if idx == 0:
                    continue
                prev_end = phy_intrvs[k][idx - 1][1]
                if prev_end > seg[0]:
                    conflict = True
                    break
            if conflict:
                continue
            # Update the segs
            for seg in segs:
                idx = bisect_left(starts[k], seg[0])
                phy_intrvs[k].insert(idx, seg)
                starts[k].insert(idx, seg[0])
            # assign
            cfg[i] = k
            allocated = True
            refill_count += 1
            break
        # spill
        if not allocated:
            cfg[i] = regnum
    cfg = F.one_hot(cfg, num_classes = regnum + 1)
    if net!= None:
        cur_obj = net.objEval(cfg)
        print("=========== Refill ==========")
        print("Refill count : %.6f, Time : %.4f"%(refill_count, time.time() - s))
        print("Before %.6f, After %.6f"%(prev_obj, cur_obj))
    return cfg

