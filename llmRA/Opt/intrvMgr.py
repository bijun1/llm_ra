import json

class SubInterval:
    def __init__(self, idx, info):
        self.prno, self.spillable, self.segs, self.allowed, self.ori, self.hint, self.hint_freq = info
        self.allowed = set(self.allowed)
        self.idx = idx
        self.reset()
    
    def _is_intersecting(self, seg1, seg2):
        start1, end1 = seg1
        start2, end2 = seg2
        return not (end1 <= start2 or end2 <= start1)


    def intersect(self, tsintrv):
        i, j = 0, 0
        while i < len(self.segs) and j < len(tsintrv.segs):
            if self._is_intersecting(self.segs[i], tsintrv.segs[j]):
                return True
            if self.segs[i][1] < tsintrv.segs[j][1]:
                i += 1
            else:
                j += 1
        return False
    
    def reset(self):
        self.phy_reg_idx = None
        self.is_spilled = False
    
    def hint_count(self):
        if self.phy_reg_idx == self.hint:
            return 0
        else:
            return self.hint_freq

class Interval:
    def __init__(self, prno):
        self.prno = prno
        self.sub_intrvs = []
        self.sprno2sintrv = {}
        self.edges = []
        self.succ_spillable_edges = []
    
    def add_sub_intrv(self, sub_intrv):
        self.sub_intrvs.append(sub_intrv)
        self.sprno2sintrv[sub_intrv.prno] = sub_intrv
    
    def add_edge(self, edge):
        fno, tno, _ = edge
        assert fno in self.sprno2sintrv
        assert tno in self.sprno2sintrv
        self.edges.append(edge)
        if self.sprno2sintrv[fno].spillable and self.sprno2sintrv[tno].spillable:
            self.succ_spillable_edges.append(edge)
    
    def set_conf_and_allowed(self, conf_graph):
        self.confs = set()
        self.allowed = set([x for x in self.sub_intrvs[0].allowed])
        for subintrv in self.sub_intrvs:
            self.confs = self.confs.union(conf_graph[subintrv.prno])
            self.allowed = self.allowed & subintrv.allowed
    
    def move_count(self):
        cost = 0
        for edge in self.edges:
            fno, tno, w = edge
            fintrv, tintrv = self.sprno2sintrv[fno], self.sprno2sintrv[tno]
            if fintrv.is_spilled or tintrv.is_spilled:
                continue
            if fintrv.phy_reg_idx != tintrv.phy_reg_idx:
                cost += w
        return cost

    def reload_count(self):
        cost = 0
        for edge in self.edges:
            fno, tno, w = edge
            fintrv, tintrv = self.sprno2sintrv[fno], self.sprno2sintrv[tno]
            if fintrv.is_spilled and not tintrv.is_spilled:
                cost += w
        return cost

    def spill_count(self):
        cost = 0
        for edge in self.edges:
            fno, tno, w = edge
            fintrv, tintrv = self.sprno2sintrv[fno], self.sprno2sintrv[tno]
            if not fintrv.is_spilled and tintrv.is_spilled:
                cost += w
        return cost
    
    def hint_count(self):
        cost = 0
        for si in self.sub_intrvs:
            cost += si.hint_count()
        return cost

class IntrvMgr:
    def __init__(self, interv_path, edge_path):
        self.interv_path = interv_path
        self.edge_path = edge_path
        self.interv_info = []
        self.edge_info = []
        
        self.sub_intrvs = []
        self.intrvs = []
        self.new_intrvs = []
        self.sub_intrv2idx = {}
        self.intrv2idx = {}

        self.phy_regs = set()

        self.callee_saved = None

        self.load_intervs()
        self.load_edges()
        self.max_prno = 0
        self.max_ori_prno = 0

    def get_si(self, prno):
        return self.sub_intrvs[self.sub_intrv2idx[prno]]

    def load_intervs(self, delimiter=';'):
        with open(self.interv_path, 'r') as f:
            lineno = 0
            for line in f:
                line = line.strip()
                if line == "]":
                    self.callee_saved = []
                    lineno += 1
                    continue
                seps = line.split(delimiter)
                line_info = []
                for v in seps:
                    if "[" not in v:
                        try:
                            line_info.append(int(v))
                        except:
                            line_info.append(float(v))
                    else:
                        line_info.append(json.loads(v))
                if lineno == 0:
                    self.callee_saved = line_info[0]
                    lineno += 1
                    continue
                segs = line_info[2]
                new_segs = []
                for i, seg in enumerate(segs):
                    start = int(seg[0])
                    end = int(seg[1])
                    if i == 0:
                        new_segs.append([start, end])
                    else:
                        if start == new_segs[-1][1]:
                            new_segs[-1][1] = end
                        else:
                            new_segs.append([start, end])
                line_info[2] = new_segs
                self.interv_info.append(line_info)
                lineno += 1

    def load_edges(self, delimiter=";"):
        with open(self.edge_path, 'r') as f:
            for line in f:
                line = line.strip()
                seps = line.split(delimiter)
                line_info = []
                for v in seps:
                    line_info.append(float(v))
                self.edge_info.append(line_info)

    def newInterval(self):
        newprno = self.max_prno + len(self.new_intrvs) + 1
        intrv = Interval(newprno)
        self.intrv2idx[newprno] = len(self.intrvs)
        self.intrvs.append(intrv)
        self.new_intrvs.append(intrv)
        return intrv
    
    def reset(self):
        to_remove = set()
        for intrv in self.new_intrvs:
            to_remove.add(self.intrv2idx[intrv.prno])
            del self.intrv2idx[intrv.prno]
        intrvs = [self.intrvs[i] for i in range(len(self.intrvs)) if i not in to_remove]
        self.intrvs = intrvs
        self.new_intrvs = []
        for subintrv in self.sub_intrvs:
            subintrv.reset()
    
    def toCFG(self):
        cfgs = []
        for si in self.sub_intrvs:
            if si.is_spilled:
                assert si.spillable
                cfgs.append(-1)
            else:
                assert si.phy_reg_idx != None
                cfgs.append(si.phy_reg_idx)
        return cfgs

    def ana_cost(self):
        move_cnts, load_cnts, store_cnts = 0, 0, 0
        for intrv in self.intrvs[:self.ori_intrvs_num]:
            mcnt, lcnt, scnt, hcnt = intrv.move_count(), intrv.reload_count(), intrv.spill_count(), intrv.hint_count()
            move_cnts += mcnt 
            load_cnts += lcnt 
            store_cnts += scnt 
            move_cnts += hcnt 
        return [move_cnts, load_cnts, store_cnts]
    
    def analyze(self):
        for i, info in enumerate(self.interv_info):
            subintrv = SubInterval(i, info)
            self.sub_intrv2idx[subintrv.prno] = i
            self.sub_intrvs.append(subintrv)
            intrv_idx = subintrv.ori
            if intrv_idx not in self.intrv2idx:
                intrv = Interval(intrv_idx)
                self.intrv2idx[intrv_idx] = len(self.intrvs)
                self.intrvs.append(intrv)
            intrv = self.intrvs[self.intrv2idx[intrv_idx]]
            intrv.add_sub_intrv(subintrv)
            self.phy_regs = self.phy_regs.union(subintrv.allowed)
            self.max_prno = max(subintrv.prno, self.max_prno)
            self.max_prno = max(intrv.prno, self.max_prno)
            self.max_ori_prno = max(intrv.prno, self.max_ori_prno)
        
        self.ori_intrvs_num = len(self.intrvs)
        self.max_freq = 0
        # Edges
        for tup in self.edge_info:
            fno, _, freq = tup
            fsubintrv = self.sub_intrvs[self.sub_intrv2idx[fno]]
            intrv = self.intrvs[self.intrv2idx[fsubintrv.ori]]
            intrv.add_edge(tup)
            self.max_freq = max(self.max_freq, freq)
