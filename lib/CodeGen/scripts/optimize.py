import os
import math
import time
import torch
import shutil
import argparse
import torch.nn as nn
import torch.optim as optim
from intervalMgr import IntvMgr
from torch_scatter import scatter_add
from legalizer import legalization, refill, legalization_v2

from torch.utils.tensorboard import SummaryWriter
from estimators import *
import networkx as nx
import matplotlib.pyplot as plt


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  # torch.use_deterministic_algorithms(True)


class Net(nn.Module):
    def __init__(self, intvmgr, args):
        super(Net, self).__init__()
        self.efrom = intvmgr.efroms.to(args.device)
        self.eto = intvmgr.etos.to(args.device)

        self.eweights = intvmgr.eweights.to(args.device)
        self.em = self.eweights.max()
        self.eweights = self.eweights / self.eweights.max()


        # Program related numbers
        self.regnum = intvmgr.getRegNum()
        self.plen = intvmgr.vplen
        self.slots = self.regnum + 1
        self.intvnum = intvmgr.getIntvNum()
        self.ori_intvnum = intvmgr.getOriIntvsNum()
        self.fouriernum = args.fouriernum
        self.bin_size = args.bin_size
        self.bin_num = intvmgr.bin_num
        self.edgenum = self.efrom.shape[0]

        # Magic numbers
        self.move_cost = 0.5
        self.spill_cost = 100
        self.tau = 0.6

        # Params
        self.param = nn.Parameter(torch.zeros(self.intvnum, self.slots))
        self.ori_param = nn.Parameter(torch.zeros(self.ori_intvnum, self.slots))

        # Check points
        # [F, BS]
        y = (torch.arange(0, self.bin_size) + 0.5).view(1, self.bin_size).repeat(self.fouriernum, 1).to(args.device)
        my = torch.arange(1, self.fouriernum + 1).view(args.fouriernum, 1).repeat(1, self.bin_size).to(args.device)
        self.check_points = torch.sin(math.pi * my * (y / self.bin_size))

        # [NS, R, F]
        self.fourier_coes = intvmgr.fourier_coes.view(-1, 1, self.fouriernum).repeat(1, self.regnum, 1).to(args.device)
        self.intrv_idxs = intvmgr.intrv_idxs.to(args.device).long()
        self.bin_idxs = intvmgr.bin_idxs.to(args.device).long()

        self.masks = intvmgr.masks.to(args.device)
        self.device = args.device
        self.ori = intvmgr.ori.to(args.device)
        self.mu = torch.ones(self.bin_num * self.bin_size).to(args.device) * 0
        self.freqs = intvmgr.freqs.to(args.device)[:self.bin_num * self.bin_size]* + 1
        self.sizes = torch.Tensor(intvmgr.sizes).to(args.device)
    

    def genSampled(self):
        ori = torch.index_select(self.ori_param, 0, self.ori)
       #self.sampled = gumbel_rao(0.1 * self.param + 0.9 * ori - 1e3 * (1 - self.masks), 10, temp = self.tau)
        self.sampled, self.s_soft = gumbel_softmax(0.9* self.param + 0.9 * ori - 1e3 * (1 - self.masks), self.tau)

    def objEval(self, cfg):
        froms = cfg[self.efrom, :]; tos = cfg[self.eto, :]
        from_allocs = froms[:, :-1]; to_allocs = tos[:, :-1]
        from_spills = froms[:, -1]; to_spills = tos[:, -1]
        move_cost = ((from_allocs - to_allocs)**2).sum(dim=-1) * self.move_cost / 2
        spill_cost = (from_spills - to_spills)**2 * (self.spill_cost - self.move_cost / 2 )
        cost_weighted = self.eweights * (move_cost  + spill_cost)
        return cost_weighted.sum(dim=-1) / self.ori_intvnum
    
    def objAna(self, cfg):
        with torch.no_grad():
            froms = cfg[self.efrom, :]; tos = cfg[self.eto, :]
            from_allocs = froms[:, :-1]; to_allocs = tos[:, :-1]
            from_spills = froms[:, -1]; to_spills = tos[:, -1]
            move_cost = ((((from_allocs - to_allocs)**2).sum(dim=-1) - (from_spills - to_spills)**2) * 1 / 2 * self.eweights * self.em).sum()
            load_cost = (torch.relu(from_spills - to_spills) * self.eweights * self.em).sum()
            store_cost = (torch.relu(to_spills - from_spills) * self.eweights * self.em).sum()
        return move_cost, load_cost, store_cost
    
    def Potential(self, cfg):
        # [bn, R, F] @ [F, bs] -> [bn, R, bs] -> [bn*bs, R]
        sampled = cfg[:, :-1].unsqueeze(2)
        sampled_selected = torch.index_select(sampled, 0, self.intrv_idxs)
        coes_masked = self.fourier_coes * sampled_selected
        funcs = scatter_add(coes_masked, index = self.bin_idxs, dim = 0)
        potential = (funcs @ self.check_points).permute(0, 2, 1).contiguous().view(-1, self.regnum)
        return potential
    
    def truePotential(self, cfg, intvmgr):
        res = torch.zeros(self.bin_num * self.bin_size, self.regnum).to(self.device)
        cfg = torch.argmax(cfg, dim = 1)
        for i in range(cfg.shape[0]):
            idx = cfg[i].item()
            if idx == intvmgr.getRegNum():
                continue
            for seg in intvmgr.getSegs(i):
                res[seg[0]:seg[1], idx] += 1
        return res
    
    def drawPotential(self, cfg, intvmgr, fname, true = False):
        if true:
            p = self.truePotential(cfg, intvmgr).cpu()
        else:
            p = self.Potential(cfg, intvmgr).cpu()
        plt.figure()
        im = plt.imshow(p, cmap = plt.cm.hot_r, vmin=0, vmax=3, aspect=p.shape[1] / p.shape[0])
        plt.colorbar(im)
        plt.savefig('fig/' + fname + '.png')
        plt.close()
    
    def verify(self, cfg, intvmgr, i =-1):
        p = self.truePotential(cfg, intvmgr)
        m = p.max().item()
        assert m <= 1

    def cLoss(self, cfg):
        self.potential = self.Potential(cfg)
        return (torch.relu(self.potential - 1.5).mean(dim=-1) * self.mu).mean()
    
    def forward(self):
        self.genSampled()
        move_cost =  self.objEval(self.sampled)
        pressure_diff = self.cLoss(self.sampled) 
        return move_cost, pressure_diff, self.sampled

    def updateMu(self, theta, i):
        if i > 200:
            coe = 1.015
        else:
            coe = 1
        with torch.no_grad():
            self.mu = torch.relu(coe * self.mu + theta * torch.relu(self.potential - 1).mean(dim=-1))
    
    def dump_hist(self, cfg, sizes, ep, fname = "sizes", limit = 1e8):
        data = []
        for i in range(cfg.shape[0]):
            if sizes[i] > limit:
                continue
            if cfg[i, -1] == 1:
                data.append(sizes[i])
        plt.hist(data, bins=100)
        plt.savefig("fig/%s_%d.png"%(fname, ep))
        plt.close()

def optimize(net, intvmgr, args):
    if os.path.exists("fig"):
        shutil.rmtree("fig")
    os.makedirs("fig")
    optimizer = optim.Adam(net.parameters(), lr = 0.1)
    results = [] 
    intvmgr.dump_freqs()
    plt.hist([x for x in intvmgr.segs if x < 10], bins=100)
    plt.savefig("fig/hist.png")
    plt.close()

    for i in range(args.gammasteps):
        for k in range(args.steps):
            move_loss, closs, cfg = net()
            loss = move_loss + closs 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('opt/mloss', move_loss, i * args.steps + k)
            writer.add_scalar('opt/closs', closs, i * args.steps + k)
            writer.add_scalar('opt/loss', loss, i * args.steps + k)
            result = (move_loss.item(), closs.item(), cfg.detach().cpu())
            results.append(result)
        if args.dump:
            print("======= Round %d ======="%i)
            print("Losses      : ", loss.detach())
            print("MLosses     : ", move_loss.detach())
            print("CLosses     : ", torch.relu(net.potential - 1.5).mean().detach())
        # Update mu
        net.updateMu(500 * math.pow((i + 1), 0.3), i)
        if i % 100 == 0:
            intvmgr.dump(cfg.argmax(dim=1), int(i // 100))
      #     intvmgr.dump_mu(net.mu.unsqueeze(1).repeat(1, 32).detach().cpu().numpy(), int(i // 10))
      #     cfg = mergeIsolated(cfg, intvmgr, True)
      #     intvmgr.dump(cfg.argmax(dim=1), int(i // 10) + 1)
       #if i == 100:
       #    net.dump_graph(cfg, intvmgr)
       #    assert 0
         #  intvmgr.drawPotential(net.potential(cfg).detach().cpu().numpy(), int(i // 100))
     #    # intvmgr.dump_spilled(cfg.argmax(dim=1), int(i // 100))
    print(len(results))
    return results[-args.topk:], net

def writeBackori(cfg, intvmgr):
    cfg = torch.argmax(cfg, dim = 1)
    back_map = [None] * cfg.shape[0]
    for i in range(cfg.shape[0]):
        reg = cfg[i]
        if reg == intvmgr.getRegNum():
            back_map[i] = -1
        else:
            back_map[i] = intvmgr.all_regs[reg]
    with open(".ori", "w") as f:
        for v in back_map:
            f.write("%d\n"%v)

def writeBack(cfg, intvmgr):
    cfg = torch.argmax(cfg, dim = 1)
    back_map = [None] * cfg.shape[0]
    for i in range(cfg.shape[0]):
        reg = cfg[i]
        if reg == intvmgr.getRegNum():
            back_map[i] = -1
        else:
            back_map[i] = intvmgr.all_regs[reg]

    with open(".result", "w") as f:
        for v in back_map:
            f.write("%d\n"%v)

def loadori(intvmgr):
    reg_num = len(intvmgr.all_regs)
    cfg = torch.zeros([len(intvmgr.interv_info), reg_num])
    with open(".ori", "r") as f:
        i = 0
        for line in f:
            v = int(line)
            if v == -1:
                cfg[i, reg_num - 1] = 1
            else:
                regid = intvmgr.all_regs.index(v)
                cfg[i, regid]=1
            i += 1
    return cfg

if __name__ == "__main__":
    fstart = time.time()
    parser = argparse.ArgumentParser(description='Gradient Based Global Register allocation.')
    parser.add_argument('--device', default = "cuda:0")
    parser.add_argument('--fname', default = "default")
    parser.add_argument('--steps', default = 20)
    parser.add_argument('--gammasteps', default = 400)
    parser.add_argument('--topk', default = 4)
    parser.add_argument('--bin_size', default = 100)

    parser.add_argument('--fouriernum', default = 50)
    parser.add_argument('--seed', default = 0)
    parser.add_argument('--name', default = "default")
    parser.add_argument('--dump', action='store_true')
    args = parser.parse_args()
    args.dump = True
    if not torch.cuda.is_available():
        args.device = "cpu"
  # setup_seed(args.seed)
    print(args)
    finterv = "." + args.fname + "_intervals"
    fedge = "." + args.fname + "_edges"
    ffreq = "." + args.fname + "_freqs"
    intvmgr = IntvMgr(finterv, fedge, ffreq)
    intvmgr.analyze(args)
    net = Net(intvmgr, args)
    net.to(args.device)

  # cfg = loadori(intvmgr).to(args.device)
  # print(net.objEval(cfg))
  # cfg1 = legalization(cfg, intvmgr)
  # print(net.objEval(cfg1))
  # cfg2 = legalization_v2(cfg, intvmgr)
  # print(net.objEval(cfg2))
  # assert(0)

    writer = SummaryWriter('logs/%f'%time.time())
    start = time.time()
    res, net = optimize(net, intvmgr, args)
    grad_time = time.time() - start
    print("Gradient time ", time.time() - start)


    start = time.time()
    explore_res = []
    for i, (obj, closs, cfg) in enumerate(res):
        ori = cfg.clone()
        cfg = legalization_v2(cfg.to(args.device), intvmgr, net)
        writer.add_scalar('post/legalize_v2', net.objEval(cfg.to(args.device)), i * 10)
        newcfg = refill(cfg, intvmgr)
        score_after = net.objEval(newcfg.to(args.device))
        writer.add_scalar('post/refill', score_after, i * 10)
        explore_res.append([obj, score_after, closs, ori, newcfg])
    leg_time = time.time() - start
    print("legalize time ", time.time() - start)
    explore_res = sorted(explore_res, key = lambda x : x[1])
    print(explore_res[0][:3])
    cfg = explore_res[0][-1]
    writeBack(cfg, intvmgr)
    print(net.objAna(cfg))
    intvmgr.dump(cfg.argmax(dim=1), args.gammasteps)
    intvmgr.dump_spilled(cfg.argmax(dim=1), args.gammasteps)
    ptime = time.time() - fstart
    print(intvmgr.getOriIntvsNum())
    with open("stats.csv", 'a') as f:
        f.write("%s,%f,%f,%f,%f,%f,%f,%f\n"%(args.fname, explore_res[0][0],
                                           explore_res[0][1], explore_res[0][2], 
                                           intvmgr.vplen,
                                           grad_time, leg_time, ptime))    
    with open("/lustre/S/bijun/stats.csv", 'a') as f:
        f.write("%s,%f,%f,%f,%f,%f,%f,%f\n"%(args.fname, explore_res[0][0],
                                           explore_res[0][1], explore_res[0][2], 
                                           intvmgr.vplen,
                                           grad_time, leg_time, ptime))    

