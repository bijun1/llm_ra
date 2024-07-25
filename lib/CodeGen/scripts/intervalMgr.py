import json
import torch
import math
import cairocffi as cairo
import matplotlib.pyplot as plt
import numpy as np
import pdb

from matplotlib import cm
from matplotlib.ticker import LinearLocator
from collections import deque

import os

class IntvMgr:
    def __init__(self, interv_path, edge_path, freqs_path):
        self.interv_path = interv_path
        self.edge_path = edge_path
        self.freqs_path = freqs_path
        self.interv_info = []
        self.edge_info = []
        self.freqs_info = []
        self.all_regs = None

        # Extracted useful imformations
        self.vplen = 0
        # Real program length
        self.fourier_coes = None
        self.efroms = None
        self.etos = None
        self.link_table = []
        self.link_freqs = []
        self.eweights = None
        self.masks = None
        self.sizes = None
        self.max_unspillable_size = 0

        self.callee_saved = None
        self.group_num = 0

        self.ori = []
        self.key_dict = {
            "prno"          : 0,
            "spillable"     : 1,
            "segs"          : 2,
            "allowed"       : 3,
            "oriprno"       : 4,
            "phint"         : 5,
            "vhint"         : 6,
            "hintfreq"      : 7,
            "parent"        : 8
        }
        self.load_intervs()
        self.load_edges()
        self.load_freqs()

    def getRegNum(self):
        return len(self.all_regs) - 1
    
    def getIntvNum(self):
        return len(self.interv_info)

    def getSegs(self, i):
        return self.interv_info[i][2]

    def getOriIntvsNum(self):
        return self.ori.max().item() + 1
    
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
                align = 4
                new_segs = []
                for i, seg in enumerate(segs):
                    assert(seg[0] % align == 0 and seg[1] % align == 0)
                    start = int(seg[0] / align)
                    end = int(seg[1] / align)
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

    def load_freqs(self, delimiter=";"):
        with open(self.freqs_path, 'r') as f:
            for line in f:
                line = line.strip()
                seps = line.split(delimiter)
                line_info = []
                for v in seps:
                    line_info.append(float(v))
                self.freqs_info.append(line_info)

    def dump_mu(self, data, ep):
        plt.figure()
        im = plt.imshow(data, cmap = plt.cm.hot_r, vmin=0, vmax=100, aspect=data.shape[1] / data.shape[0])
        plt.colorbar(im)
        plt.savefig('fig/mu_' + '{:04}'.format(ep) + '.png')
        plt.close()

    
    def dump_spilled(self, cfg, ep):
        line_width = 0.8; height = 800; width = 800; space = 50
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width + 2*space, height + 2*space)
        ctx = cairo.Context(surface)
        def draw_rect(x1, y1, x2, y2):
            ctx.move_to(x1, y1)
            ctx.line_to(x1, y2)
            ctx.line_to(x2, y2)
            ctx.line_to(x2, y1)
            ctx.close_path()
            ctx.stroke()
        ctx.set_source_rgb(1, 1, 1)
        ctx.rectangle(space, space, width, height)
        ctx.fill()
        ctx.set_line_width(line_width)
        ctx.set_source_rgba(0.1, 0.1, 0.1, alpha=0.8)
        ctx.move_to(space, space)
        ctx.line_to(space, space + height)
        ctx.line_to(space + width, space + height)
        ctx.line_to(space + width, space)
        ctx.close_path()
        ctx.stroke()
        ctx.set_line_width(line_width)
        ctx.set_source_rgba(0, 0, 0, alpha=1)
        idxs = [i for i in range(cfg.shape[0]) if cfg[i] == self.getRegNum()]
        rec_width = width / len(idxs)
        y_scale = height / self.vplen
        for idx, i in enumerate(idxs):
            segs = self.interv_info[i][2]
            for seg in segs:
                ctx.rectangle(space + idx * rec_width, space + seg[0] * y_scale, rec_width, (seg[1] - seg[0]) * y_scale)
                ctx.fill()
                rect = (space + idx * rec_width, space + seg[0] * y_scale, space + (idx + 1)*rec_width, space + seg[1] * y_scale)
                draw_rect(*rect)
        surface.write_to_png('fig/spilled' + '{:04}'.format(ep) + '.png')

    def dump(self, cfg, ep, gradient=None):
        line_width = 0.8
        height = 800
        width = 800
        space = 50
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width + 2*space, height + 2*space)
        ctx = cairo.Context(surface)
        rec_width = width / len(self.all_regs)
        y_scale = height / self.vplen

        def draw_rect(x1, y1, x2, y2):
            ctx.move_to(x1, y1)
            ctx.line_to(x1, y2)
            ctx.line_to(x2, y2)
            ctx.line_to(x2, y1)
            ctx.close_path()
            ctx.stroke()
        
        def add_text(x_center, y_center, str_):
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(10)
            ctx.set_source_rgb(1, 1, 1)
            extents = ctx.text_extents(str_)
            ctx.move_to(x_center - extents[4] / 2, y_center + extents[3] / 2)
            ctx.show_text(str_)

        ctx.set_source_rgb(1, 1, 1)
        ctx.rectangle(space, space, width, height)
        ctx.fill()
        ctx.set_line_width(line_width)
        ctx.set_source_rgba(0.1, 0.1, 0.1, alpha=0.8)
        ctx.move_to(space, space)
        ctx.line_to(space, space + height)
        ctx.line_to(space + width, space + height)
        ctx.line_to(space + width, space)
        ctx.close_path()
        ctx.stroke()
        ctx.set_line_width(line_width)
        # Draw label
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_source_rgb(1, 1, 1)
        ctx.move_to(space, space + height)
        ctx.show_text("Y")
        ctx.move_to(space + width, space)
        ctx.show_text("X")
        vs = self.vplen // 20 * y_scale
        for i in range(20):
            ctx.move_to(space + width, space + i * vs)
            ctx.show_text("%d"%(i * self.vplen // 20))

        for i in range(self.getRegNum()):
            ctx.move_to(space + (i) * rec_width, space)
            ctx.show_text("R%d"%(i))
        ctx.move_to(space + (self.getRegNum()+ 0.1) * rec_width, space)
        ctx.show_text("STACK")

        # Draw Spillable intervals
        for i in range(cfg.shape[0]):
            if self.interv_info[i][1] != 1:
                continue
            idx = cfg[i]
            segs = self.interv_info[i][2]
            for seg in segs:
                if cfg[i] != self.getRegNum():
                    ctx.set_source_rgba(1, 0, 0, alpha=0.5)
                else:
                    ctx.set_source_rgba(0, 1, 0, alpha=0.5)
                ctx.rectangle(space + idx * rec_width, space + seg[0] * y_scale, rec_width, (seg[1] - seg[0]) * y_scale)
                ctx.fill()
                rect = (space + idx * rec_width, space + seg[0] * y_scale, space + (idx + 1)*rec_width, space + seg[1] * y_scale)
                draw_rect(*rect)
               #add_text(space + (idx + 1.5) * rec_width, space + (seg[0] + seg[1]) / 2 * y_scale, "%d"%(self.interv_info[i][4]))
        
        # Draw Unspillable intervals
        rects = []
        for i in range(cfg.shape[0]):
            if self.interv_info[i][1] == 1:
                continue
            idx = cfg[i]
            segs = self.interv_info[i][2]
            for seg in segs:
                ctx.set_source_rgba(0, 0, 1, alpha=0.5)
                ctx.rectangle(space + idx * rec_width, space + seg[0] * y_scale, rec_width, (seg[1] - seg[0]) * y_scale)
                ctx.fill()
                rects.append((space + idx * rec_width, space + seg[0] * y_scale, space + (idx + 1)*rec_width, space + seg[1] * y_scale))
                add_text(space + (idx + 0.5) * rec_width, space + (seg[0] + seg[1]) / 2 * y_scale, "%d"%self.interv_info[i][4])
        ctx.set_source_rgba(0, 0, 0.8, alpha=0.5)
        for rect in rects:
            draw_rect(*rect)
        
        # Show iteration
        ctx.set_source_rgb(0,0,0)
        ctx.set_line_width(line_width * 10)
        ctx.select_font_face("monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(32)
        ctx.move_to(1, 1)
        ctx.show_text('{:04}'.format(ep))
        surface.write_to_png('fig/ep' + '{:04}'.format(ep) + '.png')

    def getIntervalStart(self, i):
        return self.getIntervInfo("segs", i)[0][0] 

    def getIntervalEnd(self, i):
        return self.getIntervInfo("segs", i)[-1][1] 

    def isSpillable(self, i):
        return self.interv_info[i][1]

    def getIntervInfo(self, key, intrv_id):
        assert key in self.key_dict.keys()
        return self.interv_info[intrv_id][self.key_dict[key]]
    
    def calcReloadAdv(self, i, cfg):
        assert len(cfg.shape) == 1
        linked = self.link_table[i]
        freqs = torch.Tensor(self.link_freqs[i]).to(cfg.device)
        alloced = cfg[linked]
        is_spilled = (alloced == self.getRegNum()).to(float)
        return ((1 - 2 * is_spilled) * freqs).sum()

    def calcMoveAdv(self, i, cfg, k):
        assert len(cfg.shape) == 1
        linked = self.link_table[i]
        freqs = torch.Tensor(self.link_freqs[i]).to(cfg.device)
        alloced = cfg[linked]
        cost_ori = ((1 - (alloced == cfg[i]).to(float)) * freqs).sum()
        cost_new = ((1 - (alloced == k).to(float)) * freqs).sum()
        return cost_ori - cost_new
    
    def dump_freqs(self):
        plt.plot(self.freqs.numpy())
        plt.savefig("fig/freq.png")
        plt.close()
    
    def analyze(self, args):
        all_regs = set()
        idx2prno = []
        ori2intvs = {}
        # PR, spillable, segs, regs, ori, vhint, phint, hintfreq
        for i, tup in enumerate(self.interv_info):
            # pr number
            idx2prno.append(self.getIntervInfo("prno", i))
            allowed = set(self.getIntervInfo("allowed", i))
            all_regs = all_regs.union(allowed)
            self.vplen = max(self.vplen, self.getIntervalEnd(i))
            orinum = self.getIntervInfo("oriprno", i)
            self.ori.append(orinum)
            if orinum not in ori2intvs:
                ori2intvs[orinum] = []
            ori2intvs[orinum].append(i)
        self.ori = torch.LongTensor(self.ori)

        # Allocable mask
        self.all_regs = sorted(list(all_regs)) + [-1]
        masks = []
        spillable_set = set()
        for i, tup in enumerate(self.interv_info):
            mask = [0] * len(self.all_regs)
            allowed = self.getIntervInfo("allowed", i)
            for k, reg in enumerate(allowed):
              # if reg not in [1,2,3,17,18,19]:
              #     continue
              # if reg in [10, 11, 12, 13, 14]:
              #     continue
                idx = self.all_regs.index(reg)
                mask[idx] = 1
            if self.getIntervInfo("spillable", i) == 1:
                spillable_set.add(i)
                mask[-1] = 1
            masks.append(mask)
            self.link_table.append([])
            self.link_freqs.append([])
        self.masks = torch.Tensor(masks)

        # Edges
        prno2idx = [None] * (max(idx2prno) + 1)
        for i, v in enumerate(idx2prno):
            prno2idx[v] = i
        efroms = []; etos = []; weights = []
        for tup in self.edge_info:
            from_id = prno2idx[int(tup[0])]
            to_id = prno2idx[int(tup[1])]
            efroms.append(from_id)
            etos.append(to_id)
            self.link_table[from_id].append(to_id)
            self.link_freqs[from_id].append(tup[2])
            self.link_table[to_id].append(from_id)
            self.link_freqs[to_id].append(tup[2])
            weights.append(tup[2])
        self.efroms = torch.LongTensor(efroms)
        self.etos = torch.LongTensor(etos)
        self.eweights = torch.Tensor(weights)

        def FSeries(start, end, T):
            mid = (start + end) / 2
            delta = (end - start) / 2
            coes = []
            for jy in range(args.fouriernum):
                coe = 2 * math.sin(math.pi * (jy + 1) * (mid / T)) * \
                          math.sin(math.pi * (jy + 1) * delta / T)
                coes.append(2 / (math.pi * (jy + 1)) * coe)
            return torch.Tensor(coes)

        # Segments in bins
        coes = []; intrv_idxs = []; bin_idxs = []
        self.sizes = []
        self.segs = []
        self.spillable_idxs = []
        self.unspillable_idxs = []
        for i, tup in enumerate(self.interv_info):
            segs = tup[2]
            bin_segs = {}
            size = 0
            self.segs.append(len(segs))
            for seg in segs:
                start = seg[0] + 1e-8; end = seg[1] - 1e-8
                size += end - start
                s_bin_id = int(start // args.bin_size)
                e_bin_id = int(end // args.bin_size)
                for binid in range(s_bin_id, e_bin_id + 1):
                    if binid == s_bin_id:
                        s = start
                    else:
                        s = binid * args.bin_size + 1e-8
                    if binid == e_bin_id:
                        e = end
                    else:
                        e = (binid + 1) * args.bin_size - 1e-8
                    if str(binid) not in bin_segs:
                        bin_segs[str(binid)] = []
                    bin_segs[str(binid)].append([s, e])
            self.sizes.append(size)
            for key in bin_segs.keys():
                segs = bin_segs[key]
                fseries = [FSeries(x[0] % args.bin_size, x[1] % args.bin_size, args.bin_size) for x in segs]
                coe = sum(fseries)
                coes.append(coe.tolist())
                intrv_idxs.append(i)
                bin_idxs.append(int(key))
        self.fourier_coes = torch.Tensor(coes)
        self.intrv_idxs = torch.Tensor(intrv_idxs)
        self.bin_idxs = torch.Tensor(bin_idxs)
        self.bin_num = int(self.bin_idxs.max().item()) + 1

        # Point freqs
        self.freqs = torch.zeros(self.vplen)
        for tup in self.freqs_info:
            s, e, f = tup
            s = int(s // 4)
            e = min(int(e // 4), self.vplen)
            self.freqs[s:e] = f
        self.freqs = self.freqs / self.freqs.max()


        


