#include "RegAllocGrad.h"
#include "AllocationOrder.h"
#include "InterferenceCache.h"
#include "LiveDebugVariables.h"
#include "RegAllocBase.h"
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
#include "llvm/CodeGen/MIRPrinter.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineModuleInfo.h"


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
#include <sstream>

#include "RegAllocGreedy.h"
#include "RegAllocScore.h"

using namespace llvm;

#define DEBUG_TYPE "regalloc"

static RegisterRegAlloc gradRegAlloc("grad", "grad register allocator", createGradRegisterAllocator);

static cl::opt<bool> NoSplitFCall("no-split-fcall", cl::Hidden, cl::desc("Disable split by function call."), cl::init(false));
static cl::opt<bool> Dump("dump-grad", cl::Hidden, cl::desc("Dump program."), cl::init(false));
static cl::opt<std::string> PathPrefix("ra-path-prefix", cl::Hidden, cl::desc("Prefix for pathes"), cl::init("./"));
static cl::opt<std::string> RAMode("ra-mode", cl::Hidden, cl::desc("Mode of regalloc : dump, tune, eval"), cl::init("dump"));

char RAGrad::ID = 0;
char &llvm::RAGradID = RAGrad::ID;

INITIALIZE_PASS_BEGIN(RAGrad, "grad", "Gradient Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveDebugVariables)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(LiveStacks)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrix)
INITIALIZE_PASS_DEPENDENCY(EdgeBundles)
INITIALIZE_PASS_END(RAGrad, "grad", "Gradient Register Allocator", false, false)

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
  AU.addRequired<LiveStacks>();
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
  for (Register virtreg : virtreg_worklist) {
    LiveInterval* intv = &LIS->getInterval(virtreg);
    Register orireg = VRM->getOriginal(virtreg);
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
        emptyIntervs.push_back(interv->reg());
        continue; 
      } 
      identifyAllowedRegs(interv->reg());
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

bool RAGrad::instWithHints(MachineInstr& MI) {
  if (!TII->isFullCopyInstr(MI)) return false;
  if (isCoalCopy(&MI)) return false;
  Register Reg1 = MI.getOperand(0).getReg();
  Register Reg2 = MI.getOperand(1).getReg();
  return Reg1.isPhysical() || Reg2.isPhysical();
}

void RAGrad::splitByFcall() {
  SmallVector<Register, 4> newvregs;
  std::set<MachineBasicBlock*> to_split_bbs;
  for (MachineBasicBlock& MBB : *MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      if (MI.isCall() || instWithHints(MI)) {
        to_split_bbs.insert(&MBB);
        break;
      } 
    }
  }
  for (Register virtreg : virtreg_worklist) {
    LiveInterval* virtreg_interv = &LIS->getInterval(virtreg);
    if (LIS->intervalIsInOneMBB(*virtreg_interv)) {
      newvregs.push_back(virtreg);
      continue;
    }
    SA->analyze(virtreg_interv);
    // Prepare split editor.
    LiveRangeEdit LREdit(virtreg_interv, newvregs, *MF, *LIS, VRM, nullptr, &DeadRemats);
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
      newvregs.push_back(virtreg);
      continue;
    }
    // We did split for some blocks.
    SmallVector<unsigned, 8> IntvMap;
    SE->finish(&IntvMap);
    // Tell LiveDebugVariables about the new ranges.
    DebugVars->splitRegister(virtreg, LREdit.regs(), *LIS);
  }
  updateIntervals(newvregs);
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
  for (Register virtreg : virtreg_worklist) {
    LiveInterval* virtreg_interv = &LIS->getInterval(virtreg);
    SA->analyze(virtreg_interv);
    LiveRangeEdit LREdit(virtreg_interv, newvregs, *MF, *LIS, VRM, nullptr, &DeadRemats);
    SE->reset(LREdit, SplitEditor::SM_Partition);
    ArrayRef<SlotIndex> Uses = SA->getUseSlots();
    if (Uses.size() <= 1) {
      // Not splitted
      newvregs.push_back(virtreg);
      spillable.insert(virtreg);
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
      newvregs.push_back(virtreg);
      spillable.insert(virtreg);
      continue ;
    }
    SmallVector<unsigned, 8> IntvMap;
    SE->finish(&IntvMap);
    DebugVars->splitRegister(virtreg, LREdit.regs(), *LIS);
    for (auto vreg : LREdit.regs()) {
      auto interv =  &LIS->getInterval(vreg);
      if (interv->empty())  {
        emptyIntervs.push_back(vreg);
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
    virtreg_interv->clear();
  }
  updateIntervals(newvregs);
  if (Dump) { LIS->dump(); }
}

void RAGrad::updateIntervals(SmallVector<Register, 4>& newvregs) {
  virtreg_worklist.clear();
  std::vector<LiveInterval*> temp_intervs;
  std::map<Register, std::vector<LiveInterval*>> subintervals;
  for (auto vreg : newvregs) {
    LiveInterval* interv = &LIS->getInterval(vreg);
    if (interv->empty()) { 
      emptyIntervs.push_back(vreg);
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
      virtreg_worklist.push_back(seg_starts[i].second->reg());
    }
  }
}

bool RAGrad::identifyAllowedRegs(Register Reg) {
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
  regs_allowed_map.insert(std::make_pair(Reg, std::move(VRegAllowed)));
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
      emptyIntervs.push_back(Reg);
      continue;
    }
    identifyAllowedRegs(Reg);
    virtreg_worklist.push_back(Reg);
  }
  updateRegMap();
}

void RAGrad::dumpScores(std::vector<std::vector<double>>& scores) {
  std::ofstream outfile;
  outfile.open(PathPrefix + "." + MF->getName().str() + "_scores");
  std::string str = "";
  for (auto & score : scores) {
    for (double v : score) {
      str.append(std::to_string(v) + ",");
    }
    str = str.substr(0, str.size() - 1);
    str.append("\n");
  }
  outfile << str;
  outfile.close();
}

void RAGrad::dumpSize() {
  std::ofstream outfile;
  outfile.open(PathPrefix + ".sizes", std::ios_base::app);
  outfile << MF->getName().str() << "," << std::to_string(virtreg_worklist.size()) << "\n";
  outfile.close();
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
  for (Register vreg : virtreg_worklist) {
    LiveInterval* interv = &LIS->getInterval(vreg);
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
    auto intv_regs_allowed = regs_allowed_map[vreg];
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
        regs_allowed_map[vreg].push_back(OtherReg);
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
  dumpSize();
  std::vector<int> config;
  makeDConfig(config);
  assignReg(config);
  finalizeAlloc();
  spiller().postOptimization();
  clearDeadRemats();
}

void RAGrad::tune() {
  std::vector<std::vector<int>> configs;  
  makeConfigs(configs);
  saveMF(*MF);
  restoreMF(*MF);
  saveMF(*MF);
  std::vector<std::vector<double>> scores;
  for (int i = 0; i < int(configs.size()); i++) {
    assignReg(configs[i]);
    finalizeAlloc();
    spiller().postOptimization();
    clearDeadRemats();
    Score score = calculateRegAllocScore();
    scores.push_back({score.getScore(), score.CopyCounts, score.LoadCounts, score.StoreCounts});
    if (i == int(configs.size() - 1)) continue;
    restoreMF(*MF);
    saveMF(*MF);
  }
  dumpScores(scores);
}

Register RAGrad::getAllowedAlias(Register virtreg, int phyidx) {
    auto intv_regs_allowed = regs_allowed_map[virtreg];
    auto regset = phy2alias[phyidx];
    for (auto reg : regset) {
      auto it = std::find(intv_regs_allowed.begin(), intv_regs_allowed.end(), reg);
      if (it == intv_regs_allowed.end()) { continue; }
      return *it;
    }
    assert(false && " Unallowable reg");
    return Register(-1);
}

// Make a default configuration by spilling all spillable intervs
void RAGrad::makeDConfig(std::vector<int>& result) {
  result.clear();
  std::map<Register, LiveInterval*> reg2interv_;
  for (Register virtreg : virtreg_worklist) {
    // Spill all spillable
    if (spillable.find(virtreg) != spillable.end()) {
      result.push_back(-1);
      continue;
    }
    // Alloc for a allowd non-conflict register
    for (auto phyreg : regs_allowed_map[virtreg]) {
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
  for (Register virtreg : virtreg_worklist) {
    LiveInterval* vreginterv = &LIS->getInterval(virtreg);
    int idx = result[i];
    i++;
    if (idx == -1)  {
      spilled.push_back(vreginterv);
      spilled_set.insert(virtreg);
      continue;
    }
    Register phyreg = getAllowedAlias(virtreg, idx);
    VRM->assignVirt2Phys(virtreg, phyreg);
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
  for (auto virtreg : emptyIntervs) {
    Register PReg = MRI.getSimpleHint(virtreg);

    if (PReg == 0) {
      const TargetRegisterClass &RC = *MRI.getRegClass(virtreg);
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

    VRM->assignVirt2Phys(virtreg, PReg);
  }
}

void RAGrad::set_ana(MachineFunction &mf) {
  TRI = &VRM->getTargetRegInfo();
  MRI = &VRM->getRegInfo();
  MRI->freezeReservedRegs(VRM->getMachineFunction());
  RegClassInfo.runOnMachineFunction(VRM->getMachineFunction());
  VRAI = new VirtRegAuxInfo(mf, *LIS, *VRM, *Loops, *MBFI);
  SpillerInstance.reset(createInlineSpiller(*this, *MF, *VRM, *VRAI));
  SA.reset(new SplitAnalysis(*VRM, *LIS, *Loops));
  SE.reset(new SplitEditor(*SA, *LIS, *VRM, *DomTree, *MBFI, *VRAI));
}

void RAGrad::get_ana(MachineFunction &mf) {
  MF = &mf;
  TII = MF->getSubtarget().getInstrInfo();
  LIS = &getAnalysis<LiveIntervals>();
  LSS = &getAnalysis<LiveStacks>();
  DomTree = &getAnalysis<MachineDominatorTree>();
  Loops = &getAnalysis<MachineLoopInfo>();
  Indexes = &getAnalysis<SlotIndexes>();
  MBFI = &getAnalysis<MachineBlockFrequencyInfo>();
  DebugVars = &getAnalysis<LiveDebugVariables>();
  VRM = &getAnalysis<VirtRegMap>();
}

void RAGrad::saveMF(MachineFunction& mf) {
  bb_states.clear();
  for (auto & BB : mf) {
    BBState state;
    state.save(BB, mf);
    bb_states.push_back(std::move(state));
  } 
  vrm_state.save(*VRM);
}

void RAGrad::restoreMF(MachineFunction& mf) {
  for (auto state : bb_states) {
    state.restore();
  }
  // VRM
  vrm_state.restore();
  // Analysis
  Indexes->releaseMemory();
  Indexes->runOnMachineFunction(mf);
  Indexes->packIndexes();
  LIS->releaseMemory();
  LIS->runOnMachineFunction(mf);
  SpillerInstance.reset(createInlineSpiller(*this, *MF, *VRM, *VRAI));
  LSS->releaseMemory();
}

void RAGrad::clearDeadRemats() {
  for (auto *DeadInst : DeadRemats) {
    LIS->RemoveMachineInstrFromMaps(*DeadInst);
    DeadInst->eraseFromParent();
  }
  DeadRemats.clear();
}

void RAGrad::makeConfigs(std::vector<std::vector<int>>& configs) {
  std::ifstream infile;
  std::string fpath = PathPrefix + "." + MF->getName().str() + "_configs";
  infile.open(fpath);
  std::string line;
  while (std::getline(infile, line)) {
    std::stringstream ss(line);
    std::string item;
    std::vector<int> config;
    while (std::getline(ss, item, ',')) {
      config.push_back(std::stoi(item));
    }
    configs.push_back(std::move(config));
  }
  infile.close();
}

Score RAGrad::calculateRegAllocScore() {
  Score score;
  for (const MachineBasicBlock &MBB : *MF) {
    double BlockFreqRelativeToEntrypoint = MBFI->getBlockFreqRelativeToEntryBlock(&MBB);
    for (const MachineInstr &MI : MBB) {
      if (MI.isDebugInstr() || MI.isKill() || MI.isInlineAsm()) {
        continue;
      }
      if (MI.isCopy()) {
        if (TII->isFullCopyInstr(MI)) {
          Register Reg1 = MI.getOperand(0).getReg();
          Register Reg2 = MI.getOperand(1).getReg();
          Register phyreg1 = Reg1.isVirtual() ? Register(VRM->getPhys(Reg1)) : Register(Reg1);
          Register phyreg2 = Reg2.isVirtual() ? Register(VRM->getPhys(Reg2)) : Register(Reg2);
          if (phyreg1 == phyreg2) continue;
        }
        score.CopyCounts += BlockFreqRelativeToEntrypoint;
      } else if (MF->getSubtarget().getInstrInfo()->isTriviallyReMaterializable(MI)) {
        if (MI.getDesc().isAsCheapAsAMove()) {
          score.CheapRematCounts += BlockFreqRelativeToEntrypoint;
        } else {
          score.ExpensiveRematCounts += BlockFreqRelativeToEntrypoint;
        }
      } else if (MI.mayLoad() && MI.mayStore()) {
        score.LoadStoreCounts += BlockFreqRelativeToEntrypoint;
      } else if (MI.mayLoad()) {
        score.LoadCounts += BlockFreqRelativeToEntrypoint;
      } else if (MI.mayStore()) {
        score.StoreCounts += BlockFreqRelativeToEntrypoint;
      }
    }
  }
  return score;
}

bool RAGrad::runOnMachineFunction(MachineFunction &mf) {
  LLVM_DEBUG(dbgs() << "********** GREEDY REGISTER ALLOCATION **********\n"
                    << "********** Function: " << mf.getName() << '\n');
  get_ana(mf);
  set_ana(mf);
  if (!hasVirtRegAlloc()) return false;
  seedLiveRegs();
  if (!NoSplitFCall) { splitByFcall(); }
  splitIntervals();
  specifyEdges();
  clearDeadRemats();
  if (RAMode == "dump") {
    dumpInfo();
  } else if (RAMode == "tune") {
    tune();
  } else if (RAMode == "eval") {
    assert(0);
  } else {
    assert(0&&"Unsupported RAMode.");
  }
  clear();
  return true;
}
