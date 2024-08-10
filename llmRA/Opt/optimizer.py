import time
import heapq
import random
from .intrvMgr import IntrvMgr
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import datetime

class Sample:
    def __init__(self, cfg):
        self.cfg = cfg
    
    def set_score(self, score, score_details = None):
        self.score = score
        self.score_details = score_details
    
    def apply(self, intvmgr):
        for si, reg in zip(intvmgr.sub_intrvs, self.cfg):
            if reg == -1:
                si.phy_reg_idx = None
                si.is_spilled = True
            else:
                si.is_spilled = False
                si.phy_reg_idx = reg

class onlineOPT:
    def __init__(self, wlkeys):
        self.wlkeys = wlkeys
        self.intrvmgrs = {}
        self.conf_graphs = {}
        self.best_cfgs = {}
        self.best_scores = {}
        self.move_coe = 0.2
        self.spill_coe = 1
        self.load_coe = 4
        for wlkey in self.wlkeys:
            s = time.time()
            intrvmgr = IntrvMgr(".%s_intervals"%wlkey, ".%s_edges"%wlkey)
            intrvmgr.analyze()
            self.intrvmgrs[wlkey] = intrvmgr
            # Establish conflict graphs, record conflict set for each sub_intrv
            conf_graph = {}
            # Sort these sub intervals
            sub_intrvs = [[x.segs[0][0], x] for x in intrvmgr.sub_intrvs]
            sub_intrvs = sorted(sub_intrvs, key=lambda x : x[0])
            sub_intrvs = [x[1] for x in sub_intrvs]
            for fid, fsintrv in enumerate(sub_intrvs):
                conf_graph[fsintrv.prno] = set()
            for fid, fsintrv in enumerate(sub_intrvs):
                for tsintrv in sub_intrvs[fid+1:]:
                    if tsintrv.segs[0][0] >= fsintrv.segs[-1][1]:
                        break
                    if fsintrv.intersect(tsintrv):
                        conf_graph[fsintrv.prno].add(tsintrv.prno)
                        conf_graph[tsintrv.prno].add(fsintrv.prno)
            # Form conf for each interval
            for intrv in intrvmgr.intrvs:
                intrv.set_conf_and_allowed(conf_graph)
            self.conf_graphs[wlkey] = conf_graph
            print("%s preprocess time %f"%(wlkey, time.time() - s))
    
    def alloc(self, subintrvs, reg, reg_states):
        for subintrv in subintrvs:
            reg_states[reg].add(subintrv.prno)
            subintrv.is_spilled = False
            subintrv.phy_reg_idx = reg
    
    def spill(self, subintrvs):
        for subintrv in subintrvs:
            subintrv.is_spilled = True
            subintrv.phy_reg_idx = None
    
    def evict(self, reg, reg_states, subintervs):
        for subinterv in subintervs:
            reg_states[reg].remove(subinterv.prno)
        self.spill(subintervs)
    
    def sat_from_states(self, reg_states, intrvs, conf_graph, intrvmgr):
        while len(intrvs) > 0:
            cur_intrv = intrvs.pop()
            conf_intrvs = cur_intrv.confs
            phy_reg_list = list(intrvmgr.phy_regs)
            random.shuffle(phy_reg_list)
            free_reg = None
            for reg in phy_reg_list:
                if reg not in cur_intrv.allowed:
                    continue
                reg_state = reg_states[reg]
                if conf_intrvs.isdisjoint(reg_state):
                    free_reg = reg
                    break
            if free_reg != None:
                # Alloc for free
                self.alloc(cur_intrv.sub_intrvs, free_reg, reg_states)
            else:
                # Randomly choose an allocable phy_reg, split cur_intrv to fit in
                random.shuffle(phy_reg_list)
                reg_to_split = None
                for reg in phy_reg_list:
                    if reg not in cur_intrv.allowed:
                        continue
                    reg_state = reg_states[reg]
                    subintrvs1 = []; subintrvs2 = []
                    for subintrv in cur_intrv.sub_intrvs:
                        if conf_graph[subintrv.prno].isdisjoint(reg_state):
                           subintrvs1.append(subintrv)
                        else:
                           subintrvs2.append(subintrv)
                    # Should not be a free register
                    assert len(subintrvs2) > 0
                    if len(subintrvs1) > 0:
                        reg_to_split = reg
                        break
                if reg_to_split != None:
                    # Can split cur interv
                    # Create a new interval with no edges
                    newintrv = intrvmgr.newInterval()
                    for si in subintrvs2:
                        newintrv.add_sub_intrv(si)
                    newintrv.set_conf_and_allowed(conf_graph)
                    index = random.randint(0, len(intrvs))
                    intrvs.insert(index, newintrv)
                    # Alloc intrvs1 into this reg
                    self.alloc(subintrvs1, reg_to_split, reg_states)
                else:
                    # Should spill cur interv
                    if len(cur_intrv.sub_intrvs) > 1 or cur_intrv.sub_intrvs[0].spillable:
                        to_spill = []
                        new_intervs = []; new_indexes = []
                        for si in cur_intrv.sub_intrvs:
                            if si.spillable:
                                to_spill.append(si)
                            else:
                                newintrv = intrvmgr.newInterval()
                                newintrv.add_sub_intrv(si)
                                newintrv.set_conf_and_allowed(conf_graph)
                                index = random.randint(0, len(intrvs))
                                intrvs.insert(index, newintrv)
                                new_intervs.append([si])
                                new_indexes.append(index)
                        self.spill(to_spill)
                    else:
                        # Evict a random conflicting subinterval for this unspillable short interval
                        si = cur_intrv.sub_intrvs[0]
                        reg = random.choice(list(si.allowed))
                        allowed_regs = list(si.allowed)
                        random.shuffle(allowed_regs)
                        reg_to_evict = None
                        for reg in allowed_regs:
                            reg_state = reg_states[reg]
                            to_evict = reg_state & conf_graph[si.prno]
                            skip = False
                            for pr in to_evict:
                                if not intrvmgr.sub_intrvs[intrvmgr.sub_intrv2idx[pr]].spillable:
                                    skip = True
                                    break
                            if skip:
                                continue
                            reg_to_evict = reg
                            break
                        assert reg_to_evict != None
                        to_evict_sis = [intrvmgr.sub_intrvs[intrvmgr.sub_intrv2idx[x]] for x in to_evict]
                        self.evict(reg_to_evict, reg_states, to_evict_sis)
                        self.alloc([si], reg_to_evict, reg_states)
        mcnts, lcnts, scnts = intrvmgr.ana_cost()
        score = self.move_coe * mcnts + self.load_coe * lcnts + self.spill_coe * scnts
        cfg = intrvmgr.toCFG()
        intrvmgr.reset()
        sample = Sample(cfg)
        sample.set_score(score, [mcnts, lcnts, scnts])
        return sample
    
    def random_satisfy(self, wlkey):
        intrvmgr = self.intrvmgrs[wlkey]
        intrvmgr.reset()
        reg_states = {}
        for reg in intrvmgr.phy_regs:
            reg_states[reg] = set()
        conf_graph = self.conf_graphs[wlkey]
        intrvs = [x for x in intrvmgr.intrvs]
        random.shuffle(intrvs)
        return self.sat_from_states(reg_states, intrvs, conf_graph, intrvmgr)
    

    def explore(self, wlkey):
        print("******** Exploring %s **********"%wlkey)
        assert wlkey in self.wlkeys
        samples = []
        log_dir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir)
        best_score = 1e16; best_cfg = None
        for i in range(100):
            for j in range(100):
                sample = self.random_satisfy(wlkey)
                samples.append(sample)
                writer.add_scalar("Data", sample.score, i * 100 + j)
                if sample.score < best_score:
                    best_score = sample.score
                    best_cfg = sample.cfg
                writer.add_scalar("Best", best_score, i * 100 + j)
        samples = sorted(samples, key = lambda x : x.score)
        print([x.score for x in samples[:2]])
        print([x.score_details for x in samples[:2]])
        cfgs = [x.cfg for x in samples[:2]]
        return cfgs

    def update(self, cfgs, scores):
        for wlkey in scores:
            best_cfg = cfgs[wlkey][0]; best_score = scores[wlkey][0]
            for cfg, score in zip(cfgs[wlkey], scores[wlkey]):
                if score < best_score:
                    best_score = score
                    best_cfg = cfg
            self.best_cfgs[wlkey] = best_cfg
            self.best_scores[wlkey] = best_score


    def dump(self):
        print(self.best_scores)

class onlineGAOPT(onlineOPT):
    def create_population(self, wlkey, size):
        samples = []
        for j in range(size):
            sample = self.random_satisfy(wlkey)
            samples.append(sample)
            if sample.score < self.best_score:
                self.best_score = sample.score
                self.best_sample = sample
            self.all_pop
            self.pop_count += 1
        return samples
    
    def select(self, population, K):
        fitnesses = [x.score for x in population]
        mfitnesses = max(fitnesses)
        fitnesses = [mfitnesses - x for x in fitnesses]
        total_fitness = sum(fitnesses)
        probabilities = [f / total_fitness for f in fitnesses]
        return random.choices(population, weights=probabilities, k=K)
    
    def allocation_mutate(self, sample, wlkey):
        ratio = 0.90
        num = len(sample.cfg)
        intvmgr = self.intrvmgrs[wlkey]
        intvmgr.reset()
        reg_states = {}
        for reg in intvmgr.phy_regs:
            reg_states[reg] = set()
        conf_graph = self.conf_graphs[wlkey]
        selected_num = max(int(ratio * num), 1)
        selected = random.sample(range(num), selected_num)
        # Update the reg states
        for x in selected:
            si = intvmgr.sub_intrvs[x]
            reg = sample.cfg[x]
            if reg == -1:
                si.is_spilled = True
                si.phy_reg_idx = None
            else:
                si.is_spilled = False
                si.phy_reg_idx = reg
                reg_states[reg].add(si.prno)
        # Form to allocate intrvs
        left = [x for x in range(num) if x not in selected]
        intrv2subintrv = {}
        for x in left:
            si = intvmgr.sub_intrvs[x]
            if si.ori not in intrv2subintrv:
                intrv2subintrv[si.ori] = intvmgr.newInterval()
            intrv2subintrv[si.ori].add_sub_intrv(si)
        intrvs = []
        for key, intrv in intrv2subintrv.items():
            intrv.set_conf_and_allowed(conf_graph)
            intrvs.append(intrv)
        random.shuffle(intrvs)
        sample = self.sat_from_states(reg_states, intrvs, conf_graph, intvmgr)
        if sample.score < self.best_score:
            self.best_score = sample.score
            self.best_sample = sample
        self.pop_count += 1
        return sample
    
    def spill_pos_mutate(self, sample_, wlkey):
        intvmgr = self.intrvmgrs[wlkey]
        intvmgr.reset()
        sample_.apply(intvmgr)
        candidate_sis = []
        for intrv in intvmgr.intrvs:
            for edge in intrv.succ_spillable_edges:
                fno, tno, _ = edge
                fsi, tsi = intvmgr.get_si(fno), intvmgr.get_si(tno)
                if fsi.is_spilled and not tsi.is_spilled:
                    candidate_sis.append(tsi)
                elif tsi.is_spilled and not fsi.is_spilled:
                    candidate_sis.append(fsi)
        if len(candidate_sis) == 0:
            return None
        ratio = 0.10
        num = len(candidate_sis)
        selected_num = max(int(ratio * num), 1)
        selected = random.sample(range(num), selected_num)
        for idx in selected:
            si = candidate_sis[idx]
            si.is_spilled = True
            si.phy_reg_idx = None

        mcnts, lcnts, scnts = intvmgr.ana_cost()
        score = self.move_coe * mcnts + self.load_coe * lcnts + self.spill_coe * scnts
        cfg = intvmgr.toCFG()
        intvmgr.reset()
        sample = Sample(cfg)
        sample.set_score(score, [mcnts, lcnts, scnts])
        if sample.score < self.best_score:
            self.best_score = sample.score
            self.best_sample = sample
        self.pop_count += 1
        return sample
        

    def explore(self, wlkey):
        print("******** Exploring %s **********"%wlkey)
        assert wlkey in self.wlkeys
        log_dir = "logs/%S/"%wlkey + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir)
        self.best_score = 1e16; self.best_sample = None; self.pop_count = 0

        all_pop = []
        prev_pop = self.create_population(wlkey, 200)
        writer.add_scalar("Best", self.best_score, self.pop_count)
        for i in range(50):
            rand_pop = self.create_population(wlkey, 100)
            writer.add_scalar("Best", self.best_score, self.pop_count)
            pop = prev_pop + rand_pop
            off_spring = []
            for j in range(200):
                selected = self.select(pop, 1)
                mutation_type = random.choice(["alloc", "spill_pos"])
                if mutation_type == "alloc":
                    child = self.allocation_mutate(selected[0], wlkey)
                elif mutation_type == "spill_pos":
                    child = self.spill_pos_mutate(selected[0], wlkey)
                    if child == None:
                        continue
                else:
                    assert 0
                off_spring.append(child)
            writer.add_scalar("Best", self.best_score, self.pop_count)
            print(self.best_sample.score)
            prev_pop = [self.best_sample] + off_spring
        samples = sorted(samples, key = lambda x : x.score)
        print([x.score for x in samples[:2]])
        print([x.score_details for x in samples[:2]])
        cfgs = [x.cfg for x in samples[:2]]
        return cfgs
