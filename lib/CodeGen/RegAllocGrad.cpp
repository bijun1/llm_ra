#include "RegAllocGrad.h"
#include "AllocationOrder.h"
#include "InterferenceCache.h"
#include "LiveDebugVariables.h"
#include "RegAllocBase.h"
#include "RegAllocEvictionAdvisor.h"
#include "RegAllocPriorityAdvisor.h"
#include "SpillPlacement.h"
#include "SplitKit.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/EdgeBundles.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervalUnion.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/Spiller.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/BlockFrequency.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <map>

#include "RegAllocGreedy.h"
#include "RegAllocScore.h"

using namespace llvm;

#define DEBUG_TYPE "regalloc"

static RegisterRegAlloc gradRegAlloc("grad", "grad register allocator", createGradRegisterAllocator);

// Have bugs
static cl::opt<bool> NoSplitLoop("no-split-loop", cl::Hidden, cl::desc("Disable split by loop."), cl::init(true));


static cl::opt<bool> NoSplitFCall("no-split-fcall", cl::Hidden, cl::desc("Disable split by function call."), cl::init(false));
static cl::opt<bool> Dump("dump-grad", cl::Hidden, cl::desc("Dump program."), cl::init(false));
static cl::opt<bool> DumpInfo("dump-info", cl::Hidden, cl::desc("Dump information for LM process."), cl::init(false));

static cl::opt<int> loadCFGID("load-cfg-id", cl::Hidden, cl::desc("Config id for loading."), cl::init(0));
static cl::opt<std::string> PathPrefix("ra-path-prefix", cl::Hidden, cl::desc("Prefix for pathes"), cl::init("./"));

char RAGrad::ID = 0;
char &llvm::RAGradID = RAGrad::ID;

INITIALIZE_PASS_BEGIN(RAGrad, "grad",
                "Gradient Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveDebugVariables)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(RegisterCoalescer)
INITIALIZE_PASS_DEPENDENCY(MachineScheduler)
INITIALIZE_PASS_DEPENDENCY(LiveStacks)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrix)
INITIALIZE_PASS_DEPENDENCY(EdgeBundles)
INITIALIZE_PASS_DEPENDENCY(SpillPlacement)
INITIALIZE_PASS_DEPENDENCY(MachineOptimizationRemarkEmitterPass)
INITIALIZE_PASS_DEPENDENCY(RegAllocEvictionAdvisorAnalysis)
INITIALIZE_PASS_DEPENDENCY(RegAllocPriorityAdvisorAnalysis)
INITIALIZE_PASS_END(RAGrad, "grad",
                "Gradient Register Allocator", false, false)

FunctionPass* llvm::createGradRegisterAllocator() {
  return new RAGrad();
}

FunctionPass* llvm::createGradRegisterAllocator(RegClassFilterFunc Ftor) {
  return new RAGrad(Ftor);
}

RAGrad::RAGrad(RegClassFilterFunc F): MachineFunctionPass(ID),  ShouldAllocateClass(F) {;}

void RAGrad::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<MachineBlockFrequencyInfo>();
  AU.addPreserved<MachineBlockFrequencyInfo>();
  AU.addRequired<LiveIntervals>();
  AU.addPreserved<LiveIntervals>();
  AU.addRequired<SlotIndexes>();
  AU.addPreserved<SlotIndexes>();
  AU.addRequired<LiveDebugVariables>();
  AU.addPreserved<LiveDebugVariables>();
  AU.addRequired<LiveStacks>();
  AU.addPreserved<LiveStacks>();
  AU.addRequired<MachineDominatorTree>();
  AU.addPreserved<MachineDominatorTree>();
  AU.addRequired<MachineLoopInfo>();
  AU.addPreserved<MachineLoopInfo>();
  AU.addRequired<VirtRegMap>();
  AU.addPreserved<VirtRegMap>();
  AU.addRequired<LiveRegMatrix>();
  AU.addPreserved<LiveRegMatrix>();
  AU.addRequired<EdgeBundles>();
  AU.addRequired<SpillPlacement>();
  AU.addRequired<MachineOptimizationRemarkEmitterPass>();
  AU.addRequired<RegAllocEvictionAdvisorAnalysis>();
  AU.addRequired<RegAllocPriorityAdvisorAnalysis>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

//===----------------------------------------------------------------------===//
//                     LiveRangeEdit delegate methods
//===----------------------------------------------------------------------===//

void RAGrad::releaseMemory() {
  SpillerInstance.reset();
}

//===----------------------------------------------------------------------===//
//                            Direct Assignment
//===----------------------------------------------------------------------===//

/// Returns true if the given \p PhysReg is a callee saved register and has not
/// been used for allocation yet.


bool RAGrad::hasVirtRegAlloc() {
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (MRI->reg_nodbg_empty(Reg))
      continue;
    const TargetRegisterClass *RC = MRI->getRegClass(Reg);
    if (!RC)
      continue;
    if (ShouldAllocateClass(*TRI, *RC))
      return true;
  }

  return false;
}

std::string getPwd() {
  std::string path = "/lustre/S/bijun/work/llm_ra/llvm-project/llvm/lib/CodeGen/";
  return path;
}

// Calculate the edge connection and frequency
void RAGrad::specifyEdges() {
  std::map<Register, std::vector<LiveInterval*>> subintervals;
  for (auto intv : all_intervs) {
    Register orireg = VRM->getOriginal(intv->reg());
    auto find = subintervals.find(orireg);
    if (find == subintervals.end()) {
      std::vector<LiveInterval*> intervs;
      intervs.push_back(intv);
      subintervals.insert(std::make_pair(orireg, std::move(intervs)));
    } else {
      find->second.push_back(intv);
    }
  }

  for (auto iter = subintervals.begin(); iter != subintervals.end(); iter++) {
    std::vector<std::pair<SlotIndex, int>> seg_starts;
    for (LiveInterval* interv : iter->second) {
      if (interv->empty()) { 
        emptyIntervs.push_back(interv);
        continue; 
      } 
      identifyAllowedRegs(interv->reg(), nullptr);
      // Prevent spilling assertion due to previous works.
      for (auto seg : interv->segments) {
        seg_starts.push_back(std::make_pair(seg.start, Register::virtReg2Index(interv->reg())));
      }
    }
    std::sort(seg_starts.begin(), seg_starts.end(), [](std::pair<SlotIndex, int> p1, std::pair<SlotIndex, int> p2) {
      return p1.first < p2.first;
    });
    for (int i = 0; i < int(seg_starts.size()) - 1; i++) {
      int from_id = seg_starts[i].second;
      int to_id = seg_starts[i + 1].second;
      // Same interval, not splitted
      if (from_id == to_id) { continue; }
      from_ids.push_back(from_id); to_ids.push_back(to_id);
      auto move_point = seg_starts[i+1].first;
      auto bb = LIS->getMBBFromIndex(move_point);
      float freq = MBFI->getBlockFreqRelativeToEntryBlock(bb); 
      edge_freqs.push_back(freq);
    }
  }
  updateRegMap();
}

void RAGrad::splitEdges() {
  SmallVector<MachineLoop *, 8> Worklist(Loops->begin(), Loops->end());
  while (!Worklist.empty()) {
    MachineLoop *CurLoop = Worklist.pop_back_val();
    MachineDomTreeNode *N = DomTree->getNode(CurLoop->getHeader());
    SmallVector<MachineDomTreeNode *, 8> node_Worklist;
    node_Worklist.push_back(N);
    while(!node_Worklist.empty()) {
      N = node_Worklist.pop_back_val();
      MachineBasicBlock *BB = N->getBlock();
      MachineLoop *loop = Loops->getLoopFor(BB);
      if (loop == nullptr) { continue; }
      // Entry
      for (MachineBasicBlock* bb : loop->getBlocks()) {
        for (MachineBasicBlock* pred : bb->predecessors()) {
          if (loop->contains(pred)) { continue; }
          if (newpreds.find(pred) != newpreds.end() || newsuccs.find(pred) != newsuccs.end()) { continue; }
          auto newbb = pred->SplitCriticalEdge(bb, *this); 
          newpreds.insert(newbb);
        }
      }
      // Latch
      for (MachineBasicBlock* bb : loop->getBlocks()) {
        for (MachineBasicBlock* succ : bb->successors()) {
          if (loop->contains(succ)) { continue; }
          if (newpreds.find(succ) != newpreds.end() || newsuccs.find(succ) != newsuccs.end()) { continue; }
          auto newbb = bb->SplitCriticalEdge(succ, *this); 
          newsuccs.insert(newbb);
        }
      }
      for (MachineDomTreeNode *Child : reverse(N->children())) {
        node_Worklist.push_back(Child);
      }
    }
  }
}

bool RAGrad::instWithHints(MachineInstr& MI) {
  if (!TII->isFullCopyInstr(MI)) return false;
  if (isCoalCopy(&MI)) return false;
  Register Reg1 = MI.getOperand(0).getReg();
  Register Reg2 = MI.getOperand(1).getReg();
  return Reg1.isPhysical() || Reg2.isPhysical();
}

void RAGrad::splitByFcall() {
  SmallVector<Register, 4> newvregs;
  std::list<LiveInterval*> new_intervs;
  std::set<MachineBasicBlock*> to_split_bbs;
  for (MachineBasicBlock& MBB : *MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      if (MI.isCall() || instWithHints(MI)) {
        to_split_bbs.insert(&MBB);
        break;
      } 
    }
  }
  for (auto VirtReg : all_intervs) {
    if (LIS->intervalIsInOneMBB(*VirtReg)) {
      new_intervs.push_back(VirtReg);
      continue;
    }
    SA->analyze(VirtReg);
    Register Reg = VirtReg->reg();
    // Prepare split editor.
    LiveRangeEdit LREdit(VirtReg, newvregs, *MF, *LIS, VRM, nullptr, &DeadRemats);
    SE->reset(LREdit, SplitEditor::SM_Partition);
    // Handle use blocks
  	ArrayRef<SplitAnalysis::BlockInfo> UseBlocks = SA->getUseBlocks();
    std::set<MachineBasicBlock*> alive_outs;
    std::set<MachineBasicBlock*> alive_ins;
    std::set<MachineBasicBlock*> alive_through;
    std::set<MachineBasicBlock*> alives;
    std::map<MachineBasicBlock*, int > bb2bi;
  	for (int i = 0; i < int(UseBlocks.size()); i++) {
      auto BI = UseBlocks[i];
      if (BI.LiveOut && BI.LiveIn) { alive_through.insert(BI.MBB); }
      if (BI.LiveIn) { alive_ins.insert(BI.MBB); }
      if (BI.LiveOut) { alive_outs.insert(BI.MBB); }
      alives.insert(BI.MBB);
      bb2bi.insert(std::make_pair(BI.MBB, i));
    }
    for (int Number : SA->getThroughBlocks().set_bits()) {
      MachineBasicBlock *MBB = VRM->getMachineFunction().getBlockNumbered(Number);
      alive_through.insert(MBB);
      alive_ins.insert(MBB);
      alive_outs.insert(MBB);
      alives.insert(MBB);
    }
    for (auto MBB : alives) {
      if (to_split_bbs.find(MBB) == to_split_bbs.end()) { continue; }
      bool livein = alive_ins.find(MBB) != alive_ins.end();
      bool liveout = alive_outs.find(MBB) != alive_outs.end();
      SlotIndex Start = Indexes->getInstructionIndex(*MBB->getFirstNonDebugInstr());
      SlotIndex Stop = SA->getLastSplitPoint(MBB);
      while(LIS->getInstructionFromIndex(Stop) == nullptr) { Stop = Stop.getPrevIndex();}
      SE->openIntv();
      SlotIndex SegStart, SegStop;
      if (livein && liveout) {
        SegStart = SE->enterIntvBefore(Start, true);
        SegStop = SE->leaveIntvBefore(Stop);
      } else if (livein) {
        auto BI = UseBlocks[bb2bi.find(MBB)->second];
        SegStart = SE->enterIntvBefore(Start, true);
        SegStop = SE->leaveIntvAfter(BI.LastInstr);
      } else if (liveout) {
        auto BI = UseBlocks[bb2bi.find(MBB)->second];
        SegStart = SE->enterIntvBefore(BI.FirstInstr, true);
        SegStop = SE->leaveIntvBefore(Stop);
      }
      SE->useIntv(SegStart, SegStop);
    }
    // No blocks were split.
    if (LREdit.empty()) {
      new_intervs.push_back(VirtReg);
      continue;
    }
    // We did split for some blocks.
    SmallVector<unsigned, 8> IntvMap;
    SE->finish(&IntvMap);
    // Tell LiveDebugVariables about the new ranges.
    DebugVars->splitRegister(Reg, LREdit.regs(), *LIS);
  }
  updateIntervals(newvregs, new_intervs);
  if (Dump) { LIS->dump(); }
}

void RAGrad::splitByLoop() {
  SmallVector<Register, 4> newvregs;
  std::list<LiveInterval*> new_intervs;
  for (auto VirtReg : all_intervs) {
    if (LIS->intervalIsInOneMBB(*VirtReg)) {
      new_intervs.push_back(VirtReg);
      continue;
    }

    SA->analyze(VirtReg);
    Register Reg = VirtReg->reg();

    // Prepare split editor.
    LiveRangeEdit LREdit(VirtReg, newvregs, *MF, *LIS, VRM, nullptr, &DeadRemats);
    SE->reset(LREdit, SplitEditor::SM_Partition);

    // Split at loop boundaries, move codes inserted at newly created bbs
    // BB in the same loop are mapped to the same interv
  	ArrayRef<SplitAnalysis::BlockInfo> UseBlocks = SA->getUseBlocks();
    std::map<MachineLoop*, unsigned> loop2id;
    loop2id.insert(std::make_pair(nullptr, 0));
    std::set<unsigned> done;
    // Use Blocks are not new bbs
    for (auto BI : UseBlocks) {
      unsigned Number = BI.MBB->getNumber();
      unsigned IntvIn = 0, IntvOut = 0;
      SlotIndex IntfIn, IntfOut;
      MachineLoop* cur_loop = Loops->getLoopFor(BI.MBB);
      // Get or create an interv for cur_loop
      unsigned loop_interv = 0;
      auto find = loop2id.find(cur_loop);
      if (find == loop2id.end()) {
          loop_interv = SE->openIntv();
          loop2id.insert(std::make_pair(cur_loop, loop_interv));
      } else {
        loop_interv = find->second;
      }
      if (BI.LiveIn) IntvIn = loop_interv;
      if (BI.LiveOut) IntvOut = loop_interv;
      if (!IntvIn && !IntvOut) continue;
      if (IntvIn && IntvOut)
        SE->splitLiveThroughBlock(Number, IntvIn, IntfIn, IntvOut, IntfOut);
      else if (IntvIn)
        SE->splitRegInBlock(BI, IntvIn, IntfIn);
      else if (IntvOut)
        SE->splitRegOutBlock(BI, IntvOut, IntfOut);
      done.insert(Number);
    }

    // Live through
    for (int Number : SA->getThroughBlocks().set_bits()) {
      if (done.find(Number) != done.end()) continue;
      MachineBasicBlock* bb = MF->getBlockNumbered(Number);
      unsigned IntvIn = 0, IntvOut = 0;
      SlotIndex IntfIn, IntfOut;
      // Get or create an interv for cur_loop
      MachineLoop* cur_loop = Loops->getLoopFor(bb);
      unsigned loop_interv = 0;
      auto find = loop2id.find(cur_loop);
      if (find == loop2id.end()) {
          loop_interv = SE->openIntv();
          loop2id.insert(std::make_pair(cur_loop, loop_interv));
      } else {
        loop_interv = find->second;
      }
      if (newpreds.find(bb) != newpreds.end()) {
        MachineLoop* loop2 = Loops->getLoopFor(bb->getSingleSuccessor());
        auto find2 = loop2id.find(loop2);
        if (find2 == loop2id.end()) {
            IntvOut = SE->openIntv();
            loop2id.insert(std::make_pair(loop2, IntvOut));
        } else {
          IntvOut = find2->second;
        }
        IntvIn = loop_interv; 
      } else if (newsuccs.find(bb) != newsuccs.end()) {
        MachineLoop* loop2 = Loops->getLoopFor(bb->getSinglePredecessor());
        auto find2 = loop2id.find(loop2);
        if (find2 == loop2id.end()) {
            IntvIn = SE->openIntv();
            loop2id.insert(std::make_pair(loop2, IntvIn));
        } else {
          IntvIn = find2->second;
        }
        IntvOut = loop_interv; 
      } else {
        IntvOut = loop_interv;
        IntvIn = loop_interv;
      }
      if (!IntvIn && !IntvOut) continue;
      SE->splitLiveThroughBlock(Number, IntvIn, IntfIn, IntvOut, IntfOut);
    }
    
    // No blocks were split.
    if (LREdit.empty()) {
      new_intervs.push_back(VirtReg);
      continue;
    }
    // We did split for some blocks.
    SmallVector<unsigned, 8> IntvMap;
    SE->finish(&IntvMap);
    // Tell LiveDebugVariables about the new ranges.
    DebugVars->splitRegister(Reg, LREdit.regs(), *LIS);
  }
  updateIntervals(newvregs, new_intervs);
  if (Dump) { LIS->dump(); }
}

bool RAGrad::isCoalCopy(const MachineInstr* MI) {
    auto DestSrc = TII->isCopyInstr(*MI);
    if (!DestSrc)
      return false;

    const MachineOperand *DestRegOp = DestSrc->Destination;
    const MachineOperand *SrcRegOp = DestSrc->Source;
    Register dstreg =  DestRegOp->getReg();
    Register srcreg =  SrcRegOp->getReg();
    if (!dstreg.isVirtual() || !srcreg.isVirtual()) { return false; }
    return VRM->getOriginal(dstreg) == VRM->getOriginal(srcreg);
}

// Split each interval into small pieces.
// Insert copy infront of each use point.
void RAGrad::splitIntervals() {
  SmallVector<Register, 4> newvregs;
  std::list<LiveInterval*> new_intervs;
  for (auto VirtReg : all_intervs) {
    SA->analyze(VirtReg);
    LiveRangeEdit LREdit(VirtReg, newvregs, *MF, *LIS, VRM, nullptr, &DeadRemats);
    SE->reset(LREdit, SplitEditor::SM_Partition);
    ArrayRef<SlotIndex> Uses = SA->getUseSlots();
    if (Uses.size() <= 1) {
      // Not splitted
      new_intervs.push_back(VirtReg);
      spillable.insert(VirtReg->reg());
      continue ;
    }
    // Split around uses
    for (int i = 0; i < int(Uses.size()); i++) {
      auto Use = Uses[i];
      const MachineInstr *MI = Indexes->getInstructionFromIndex(Use);
      if (isCoalCopy(MI)) {continue;}
      SE->openIntv();
      SlotIndex SegStart = SE->enterIntvBefore(Use, true);
      SlotIndex SegStop = SE->leaveIntvAfter(Use, true);
      SE->useIntv(SegStart, SegStop);
    }
    if (LREdit.empty()) {
      // Not splitted
      new_intervs.push_back(VirtReg);
      spillable.insert(VirtReg->reg());
      continue ;
    }
    SmallVector<unsigned, 8> IntvMap;
    SE->finish(&IntvMap);
    DebugVars->splitRegister(VirtReg->reg(), LREdit.regs(), *LIS);
    for (auto vreg : LREdit.regs()) {
      auto interv =  &LIS->getInterval(vreg);
      if (interv->empty())  {
        emptyIntervs.push_back(interv);
        continue;
      }
      auto start = interv->segments[0].start;
      auto end = interv->segments[0].end;
      bool unspillable = false;
      for (auto use : Uses) {
        const MachineInstr *MI = Indexes->getInstructionFromIndex(use);
        if (isCoalCopy(MI)) {continue;}
        if (start <= use  && use <= end) {
          unspillable = true;
          break;
        }
      }
      if (!unspillable) { spillable.insert(vreg); }
    }
    VirtReg->clear();
  }
  updateIntervals(newvregs, new_intervs);
  if (Dump) { LIS->dump(); }
}

void RAGrad::updateIntervals(SmallVector<Register, 4>& newvregs, std::list<LiveInterval*>& newintervs) {
  all_intervs.clear();
  std::vector<LiveInterval*> temp_intervs;
  std::map<Register, std::vector<LiveInterval*>> subintervals;
  for (auto interv : newintervs) {
    if (interv->empty()) { 
      emptyIntervs.push_back(interv);
      continue;
     }
    interv->setWeight(0);
    temp_intervs.push_back(interv);
  }
  for (Register reg : newvregs) {
    auto interv = &LIS->getInterval(reg);
    if (interv->empty()) { 
      emptyIntervs.push_back(interv);
      continue;
    }
    interv->setWeight(0);
    temp_intervs.push_back(interv);
  }
  // Sort each sub interval
  for (auto intv : temp_intervs) {
    Register orireg = VRM->getOriginal(intv->reg());
    auto find = subintervals.find(orireg);
    if (find == subintervals.end()) {
      std::vector<LiveInterval*> intervs;
      intervs.push_back(intv);
      subintervals.insert(std::make_pair(orireg, std::move(intervs)));
    } else {
      find->second.push_back(intv);
    }
  }
  for (auto iter = subintervals.begin(); iter != subintervals.end(); iter++) {
    std::vector<std::pair<SlotIndex, LiveInterval*>> seg_starts;
    for (LiveInterval* interv : iter->second) {
      assert (!interv->empty()); 
      seg_starts.push_back(std::make_pair(interv->segments[0].start, interv));
    }
    std::sort(seg_starts.begin(), seg_starts.end(), [](std::pair<SlotIndex, LiveInterval*> p1, std::pair<SlotIndex, LiveInterval*> p2) {
      return p1.first < p2.first;
    });
    for (int i = 0; i < int(seg_starts.size()); i++) {
      all_intervs.push_back(seg_starts[i].second);
    }
  }
}

bool RAGrad::identifyAllowedRegs(Register Reg, std::vector<LiveInterval*>* generated) {
  const TargetRegisterClass *TRC = MRI->getRegClass(Reg);
  regs_classes.insert(TRC);
  auto VRegLI = &LIS->getInterval(Reg);
  // Record any overlaps with regmask operands.
  BitVector RegMaskOverlaps;
  LIS->checkRegMaskInterference(*VRegLI, RegMaskOverlaps);

  // Compute an initial allowed set for the current vreg.
  std::vector<MCRegister> VRegAllowed;
  ArrayRef<MCPhysReg> RawPRegOrder = TRC->getRawAllocationOrder(*MF);
  for (MCPhysReg R : RawPRegOrder) {
    MCRegister PReg(R);
    if (MRI->isReserved(PReg))
      continue;

    // vregLI crosses a regmask operand that clobbers preg.
    if (!RegMaskOverlaps.empty() && !RegMaskOverlaps.test(PReg))
      continue;

    // vregLI overlaps fixed regunit interference.
    bool Interference = false;
    for (MCRegUnit Unit : TRI->regunits(PReg)) {
      if (VRegLI->overlaps(LIS->getRegUnit(Unit))) {
        Interference = true;
        break;
      }
    }
    if (Interference)
      continue;

    // preg is usable for this virtual register.
    VRegAllowed.push_back(PReg);
  }

  // Check for vregs that have no allowed registers. These should be
  // pre-spilled and the new vregs added to the worklist.
  if (VRegAllowed.empty()) {
	  assert(generated != nullptr);
    SmallVector<Register, 8> NewVRegs;
    LiveRangeEdit LRE(VRegLI, NewVRegs, *MF, *LIS, VRM, nullptr, &DeadRemats);
    SpillerInstance->spill(LRE);
	  for (auto vreg : NewVRegs) {
	  	auto newli = &LIS->getInterval(vreg);
	  	assert(!newli->empty());
      newli->setWeight(0);
	  	generated->push_back(newli);
	  }
	  return true;
  }
  regs_allowed_map.insert(std::make_pair(VRegLI, std::move(VRegAllowed)));
  return false;
}

void RAGrad::updateRegMap() {
  // Process Register information
  for (auto rc : regs_classes) {
    ArrayRef<MCPhysReg> RawPRegOrder = rc->getRawAllocationOrder(*MF);
    for (MCPhysReg R : RawPRegOrder) {
      MCRegister PReg(R);
      bool conflict = false;
      for (int i =0 ; i < int(phy2alias.size()); ++i) {
        auto regset = &phy2alias[i];
        for (auto reg : *regset) {
          if (TRI->regsOverlap(PReg, reg)) {
            regset->insert(PReg);
            conflict = true;
            alias2phy.insert(std::make_pair(PReg, i));
            break;
          }
          if (conflict) { break; }
        } 
      }
      if (!conflict) {
        std::set<MCRegister> new_regset;
        new_regset.insert(PReg);
        alias2phy.insert(std::make_pair(PReg, phy2alias.size()));
        phy2alias.push_back(std::move(new_regset));
      }
    }
  } 
}

void RAGrad::seedLiveRegs() {
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    Register Reg = Register::index2VirtReg(i);
	  if (Reg.id() == 0) { continue; }
    if (MRI->reg_nodbg_empty(Reg))
      continue;
    LiveInterval* VRegLI = &LIS->getInterval(Reg);
    if (VRegLI->empty()) {
      emptyIntervs.push_back(VRegLI);
      continue;
    }

	  std::vector<LiveInterval*> generated;
    bool spilled = identifyAllowedRegs(Reg, &generated);
	  if (!spilled) {
      all_intervs.push_back(VRegLI);
	  } else {
		  all_intervs.insert(all_intervs.end(), generated.begin(), generated.end());
	  }
  }
  updateRegMap();
}

void RAGrad::dumpIntervals() {
  std::ofstream outfile;
  outfile.open(PathPrefix + "." + MF->getName().str() + "_intervals");
  // For callee saved registers
  std::string cs_str = "[";
  if (TRI->getCSRFirstUseCost() > 0 ) {
    for (int i = 0; i < int(phy2alias.size()); i++) {
      auto regset = phy2alias[i];
      bool is_saved = false;
      for (auto reg : regset) {
        if (!isACalleeSavedRegister(reg)) { continue; } 
        is_saved = true;
        break;
      }
      if (is_saved) { cs_str.append(std::to_string(i) + ",");}
    }
    cs_str = cs_str.substr(0, cs_str.size() - 1);
  }
  outfile << cs_str << "]\n";
  // Intervals
  for (auto interv : all_intervs) {
    Register vreg = interv->reg();
    assert(vreg.isVirtual());
    int regid = Register::virtReg2Index(vreg);
    bool canspill = (spillable.find(vreg) != spillable.end());
    // Regid, Spillable
    outfile << regid << ';' << canspill << ";[";
    // Segments
    for (int i = 0; i < int(interv->segments.size()); i++) {
      auto seg = interv->segments[i];
      auto start = seg.start.val();
      auto end = seg.end.val();
      if (i != int(interv->segments.size()) - 1) {
        outfile << '[' << start << ',' << end <<  "],";
      } else {
        outfile << '[' << start << ',' << end <<  "]";
      }
    }
    outfile << "];";
    // Reg allowed
    std::string reg_str = "[";
    auto intv_regs_allowed = regs_allowed_map[interv];
    for (auto reg : intv_regs_allowed) {
      auto find = alias2phy.find(reg);
      assert(find != alias2phy.end());
      reg_str.append(std::to_string(find->second));
      reg_str.append(",");
    }
    reg_str = reg_str.substr(0, reg_str.size() - 1);
    reg_str.append("];");
    outfile << reg_str;
    outfile << Register::virtReg2Index(VRM->getOriginal(vreg)) << ";";
    // Register Hints
    int phint = -1;
    float freq = 0;
    for (const MachineInstr &Instr : MRI->reg_nodbg_instructions(vreg)) {
      if (!TII->isFullCopyInstr(Instr)) continue;
      // Copy from one sub interval to another, already exists
      if (isCoalCopy(&Instr)) continue;
      Register OtherReg = Instr.getOperand(0).getReg();
      if (OtherReg == vreg) {
        OtherReg = Instr.getOperand(1).getReg();
        if (OtherReg == vreg) continue;
      }
      if (OtherReg.isPhysical()) {
        auto find = alias2phy.find(OtherReg);
        if (find == alias2phy.end()) { continue; }
        phint = find->second;
        regs_allowed_map[interv].push_back(OtherReg);
      }
      freq = getPointFreq(LIS->getInstructionIndex(Instr));
    }
    // Physical Hint
    outfile << phint << ";";
    outfile << freq << "\n";
  }
  outfile.close(); 
}

void RAGrad::dumpEdges() {
  std::ofstream outfile;
  outfile.open(PathPrefix + "." + MF->getName().str() + "_edges");
  for (int i = 0; i < int(from_ids.size()); i++) {
    outfile << from_ids[i] << ";" << to_ids[i] << ";"  << edge_freqs[i] << "\n";
  }
  outfile.close(); 
}

float RAGrad::getPointFreq(SlotIndex index) {
    MachineBasicBlock* bb = nullptr;
    if (MachineInstr *MI = LIS->getInstructionFromIndex(index))
      bb = MI->getParent();
    else {
      auto I = std::prev(LIS->getSlotIndexes()->getMBBUpperBound(index));
      bb = I->second;
    }
    return MBFI->getBlockFreqRelativeToEntryBlock(bb); 
}

void RAGrad::dumpInfo() {
  dumpIntervals();
  dumpEdges();
}

Register RAGrad::getAllowedAlias(LiveInterval* interv, int phyidx) {
    assert(regs_allowed_map.find(interv) != regs_allowed_map.end());
    auto intv_regs_allowed = regs_allowed_map[interv];
    auto regset = phy2alias[phyidx];
    for (auto reg : regset) {
      auto it = std::find(intv_regs_allowed.begin(), intv_regs_allowed.end(), reg);
      if (it == intv_regs_allowed.end()) { continue; }
      return *it;
    }
    assert(false && " Unallowable reg");
    return Register(-1);
}

void RAGrad::loadConfig(std::vector<int>& result) {
  std::ifstream infile;
  std::string fpath = PathPrefix + "." + MF->getName().str() + "_result_" + std::to_string(loadCFGID);
  infile.open(fpath);
  std::string line;
  while (std::getline(infile, line)) {
    std::cout << line << std::endl;
    int pos = std::stoi(line);
    result.push_back(pos);
  }
  infile.close();
}

// Make a default configuration by spilling all spillable intervs
void RAGrad::makeDConfig(std::vector<int>& result) {
  std::map<Register, LiveInterval*> reg2interv_;
  for (auto vreginterv : all_intervs) {
    // Spill all spillable
    if (spillable.find(vreginterv->reg()) != spillable.end()) {
      result.push_back(-1);
      continue;
    }
    // Alloc for a allowd non-conflict register
    for (auto phyreg : regs_allowed_map[vreginterv]) {
      auto find = reg2interv_.find(phyreg);
      // Skip non free regs
      if (find != reg2interv_.end()) continue;
      // Alloc a free reg
      bool find_reg_id = false;
      for (int phyid = 0; phyid < int(phy2alias.size()); phyid++) {
        auto regset = phy2alias[phyid];
        if (regset.find(phyreg) == regset.end()) continue;
        result.push_back(phyid);
        find_reg_id = true;
        break;
      }
      assert(find_reg_id);
      break;
    }
  }
}


void RAGrad::assignReg(std::vector<int>& result) {
  // First spill intervals, this will introduce renamed vregs
  int i = 0;
  std::list<LiveInterval*> spilled;
  std::set<Register> spilled_set;
  for (auto vreginterv : all_intervs) {
    int idx = result[i];
    i++;
    if (idx == -1)  {
      spilled.push_back(vreginterv);
      spilled_set.insert(vreginterv->reg());
      continue;
    }
    Register phyreg = getAllowedAlias(vreginterv, idx);
    VRM->assignVirt2Phys(vreginterv->reg(), phyreg);
  }
  while (spilled.size() != 0) {
    LiveInterval* vreginterv = spilled.front();
    spilled.pop_front();
    if (vreginterv->empty()) { continue; }
    vreginterv->setWeight(0);
    SmallVector<Register, 4> newvregs;
    LiveRangeEdit LRE(vreginterv, newvregs, *MF, *LIS, VRM, nullptr, &DeadRemats);
    SpillerInstance->spill(LRE, &spilled_set);
    // Spilling may rename registers
    for (auto newvreg : newvregs) {
      if (!newvreg.isVirtual()) continue;
      auto phyreg = VRM->getPhys(newvreg);
      if (phyreg != VirtRegMap::NO_PHYS_REG) { continue; }
      Register oldvreg = LRE.getVRegFromMap(newvreg);
      // Case 1. Old reg was assigned, then assign the same register
      auto oldphyreg = VRM->getPhys(oldvreg);
      if (oldphyreg != VirtRegMap::NO_PHYS_REG) {
        VRM->assignVirt2Phys(newvreg, oldphyreg);
        continue;
      }
      // Case 2. Old reg was rematerialized, and the consumer interval is spillable with reg assigned.
      // vreg1 <-- remat
      //  reg2 <--- vreg1     Assign vreg1 with reg2 to eliminate this inst
      if (LRE.isRematReg(newvreg)) {
        for (MachineInstr &MI : llvm::make_early_inc_range(MRI->reg_bundles(newvreg))) {
          if (MI.isDebugValue()) continue;
          // Find the use.
          if (!isCoalCopy(&MI)) continue;
          auto DestSrc = TII->isCopyInstr(MI);
          const MachineOperand *DestRegOp = DestSrc->Destination;
          Register dstreg =  DestRegOp->getReg();
          // Not rematerialized, ignore
          if (dstreg == newvreg) continue;
          auto dstphyreg = VRM->getPhys(dstreg);
          // Consumer also spilled, ignore them
          if (dstphyreg == VirtRegMap::NO_PHYS_REG) { continue; }
          // Same register with consumer, eliminate redundant copy
          if (VRM->getPhys(newvreg) == VirtRegMap::NO_PHYS_REG) {
            VRM->assignVirt2Phys(newvreg, dstphyreg);
          } else {
            assert(VRM->getPhys(newvreg) == dstphyreg);
          }
        }
        if (VRM->getPhys(newvreg) != VirtRegMap::NO_PHYS_REG) { continue; }
      }
      if (LIS->getInterval(newvreg).empty()) { continue; }

  //  assert(false);
    //// Case 3. Old reg is to be spilled, but splitted by LRE.
      auto finditer = std::find(spilled.begin(), spilled.end(), &LIS->getInterval(oldvreg));
      if (finditer != spilled.end() || oldvreg == vreginterv->reg()) {
        spilled.push_back(&LIS->getInterval(newvreg));
        spilled_set.insert(newvreg);
      }
    }
  }
}

bool RAGrad::isACalleeSavedRegister(MCRegister Reg) {
  const MCPhysReg *CSR = MF->getRegInfo().getCalleeSavedRegs();
  for (unsigned i = 0; CSR[i] != 0; ++i)
    if (TRI->regsOverlap(Reg, CSR[i]))
      return true;
  return false;
}

void RAGrad::finalizeAlloc() {
  MachineRegisterInfo &MRI = MF->getRegInfo();

  // First allocate registers for the empty intervals.
  for (auto LI : emptyIntervs) {
    Register PReg = MRI.getSimpleHint(LI->reg());

    if (PReg == 0) {
      const TargetRegisterClass &RC = *MRI.getRegClass(LI->reg());
      const ArrayRef<MCPhysReg> RawPRegOrder = RC.getRawAllocationOrder(*MF);
      for (MCRegister CandidateReg : RawPRegOrder) {
        if (!VRM->getRegInfo().isReserved(CandidateReg)) {
          PReg = CandidateReg;
          break;
        }
      }
      assert(PReg &&
             "No un-reserved physical registers in this register class");
    }

    VRM->assignVirt2Phys(LI->reg(), PReg);
  }
}

void RAGrad::set_ana(MachineFunction &mf) {
  Indexes->releaseMemory();
  Indexes->runOnMachineFunction(mf);
  Indexes->packIndexes();

  TRI = &VRM->getTargetRegInfo();
  MRI = &VRM->getRegInfo();
  MRI->freezeReservedRegs(VRM->getMachineFunction());

  LIS->releaseMemory();
  LIS->runOnMachineFunction(mf);


  RegClassInfo.runOnMachineFunction(VRM->getMachineFunction());
  MBFI->runOnMachineFunction(mf);
  VRAI = new VirtRegAuxInfo(mf, *LIS, *VRM, *Loops, *MBFI);
  SpillerInstance.reset(createInlineSpiller(*this, *MF, *VRM, *VRAI));
  SA.reset(new SplitAnalysis(*VRM, *LIS, *Loops));
  SE.reset(new SplitEditor(*SA, *LIS, *VRM, *DomTree, *MBFI, *VRAI));
}

void RAGrad::get_ana(MachineFunction &mf) {
  MF = &mf;
  TII = MF->getSubtarget().getInstrInfo();
  LIS = &getAnalysis<LiveIntervals>();
  DomTree = &getAnalysis<MachineDominatorTree>();
  Loops = &getAnalysis<MachineLoopInfo>();
  Indexes = &getAnalysis<SlotIndexes>();
  MBFI = &getAnalysis<MachineBlockFrequencyInfo>();
  DebugVars = &getAnalysis<LiveDebugVariables>();
  VRM = &getAnalysis<VirtRegMap>();
}

bool RAGrad::runOnMachineFunction(MachineFunction &mf) {
  LLVM_DEBUG(dbgs() << "********** GREEDY REGISTER ALLOCATION **********\n"
                    << "********** Function: " << mf.getName() << '\n');
  get_ana(mf);
  if (Dump) { LIS->dump(); }
  if (!NoSplitLoop) splitEdges();
  if (Dump) { LIS->dump(); }
  set_ana(mf);
  if (!hasVirtRegAlloc()) return false;
  seedLiveRegs();

  if (!NoSplitLoop) { splitByLoop(); }
  if (!NoSplitFCall) { splitByFcall(); }
  splitIntervals();
  specifyEdges();

  std::vector<int> result;
  dumpInfo();
  if (DumpInfo) {
    makeDConfig(result);
  } else {
    loadConfig(result);
  }
  assignReg(result);
  finalizeAlloc();
  spiller().postOptimization();
  for (auto *DeadInst : DeadRemats) {
    LIS->RemoveMachineInstrFromMaps(*DeadInst);
    DeadInst->eraseFromParent();
  }
  DeadRemats.clear();
  clear();
  if (Dump) { LIS->dump(); }
  MF->verify(this, "After register allocator");
  MBFI->runOnMachineFunction(mf);
  return true;
}
