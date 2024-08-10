#ifndef LLVM_CODEGEN_REGALLOCGRAD_H_
#define LLVM_CODEGEN_REGALLOCGRAD_H_

#include "InterferenceCache.h"
#include "RegAllocBase.h"
#include "RegisterCoalescer.h"
#include "SpillPlacement.h"
#include "SplitKit.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/Spiller.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <queue>
#include <utility>

namespace llvm {
class AllocationOrder;
class AnalysisUsage;
class EdgeBundles;
class LiveDebugVariables;
class LiveIntervals;
class LiveRegMatrix;
class MachineBasicBlock;
class MachineBlockFrequencyInfo;
class MachineDominatorTree;
class MachineLoop;
class MachineLoopInfo;
class MachineOptimizationRemarkEmitter;
class MachineOptimizationRemarkMissed;
class SlotIndexes;
class TargetInstrInfo;
class VirtRegMap;
class VirtRegAuxInfo;

class RAGrad : public MachineFunctionPass {
protected:
  const TargetRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  VirtRegMap *VRM = nullptr;
  LiveIntervals *LIS = nullptr;
  LiveRegMatrix *Matrix = nullptr;
  VirtRegAuxInfo *VRAI = nullptr;
  RegisterClassInfo RegClassInfo;
  const RegClassFilterFunc ShouldAllocateClass;
public:
  LiveIntervals *getLiveIntervals() const { return LIS; }
  VirtRegMap *getVirtRegMap() const { return VRM; }
  const RegisterClassInfo &getRegClassInfo() const { return RegClassInfo; }
  void setAnalysis(MachineFunction& mf);

private:
  // All the intervals to alloc
  std::vector<LiveInterval*> all_intervs;
  std::set<MCRegister> spillable;
  // Physical registers to aliases
  std::vector<std::set<MCRegister>> phy2alias;
  // Aliases to physical registers
  std::map<MCRegister, int> alias2phy;
  // Allowed registers for each interval
  std::map<LiveInterval*, std::vector<MCRegister>> regs_allowed_map;
  std::map<LiveInterval*, int> intrv2parent;
  // All refered register classes
  std::set<const TargetRegisterClass*> regs_classes;
  // Record all empty intervals
  std::vector<LiveInterval*> emptyIntervs;
  std::set<MachineBasicBlock*> newpreds;
  std::set<MachineBasicBlock*> newsuccs;

  // Edges from, O(n) complexity
  std::vector<int> from_ids;

  // Edges to, O(n) complexity
  std::vector<int> to_ids;

  // Edge frequencies
  std::vector<float> edge_freqs;

  /// Inst which is a def of an original reg and whose defs are already all
  /// dead after remat is saved in DeadRemats. The deletion of such inst is
  /// postponed till all the allocations are done, so its remat expr is
  /// always available for the remat of all the siblings of the original reg.
  SmallPtrSet<MachineInstr *, 32> DeadRemats;

  // context
  MachineFunction *MF = nullptr;

  // Shortcuts to some useful interface.
  const TargetInstrInfo *TII = nullptr;

  // analyses
  SlotIndexes *Indexes = nullptr;
  MachineBlockFrequencyInfo *MBFI = nullptr;
  MachineDominatorTree *DomTree = nullptr;
  MachineLoopInfo *Loops = nullptr;
  MachineOptimizationRemarkEmitter *ORE = nullptr;
  EdgeBundles *Bundles = nullptr;
  SpillPlacement *SpillPlacer = nullptr;
  LiveDebugVariables *DebugVars = nullptr;
  std::unique_ptr<SplitAnalysis> SA;
  std::unique_ptr<SplitEditor> SE;

  // state
  std::unique_ptr<Spiller> SpillerInstance;


public:
  RAGrad(const RegClassFilterFunc F = allocateAllRegClasses);

  /// Return the pass name.
  StringRef getPassName() const override { return "Grad Register Allocator"; }

  /// RAGrad analysis usage.
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override;

  Spiller &spiller() { return *SpillerInstance; }

  void clear() {
	all_intervs.clear();
	spillable.clear();
	phy2alias.clear();
	alias2phy.clear();
	regs_allowed_map.clear();
  intrv2parent.clear();
	regs_classes.clear();
	from_ids.clear();
	to_ids.clear();
	edge_freqs.clear();
  emptyIntervs.clear();
  newpreds.clear();
  newsuccs.clear();
  }

  /// Perform register allocation.
  bool runOnMachineFunction(MachineFunction &mf) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoPHIs);
  }

  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::IsSSA);
  }

  static char ID;

private:
  bool identifyAllowedRegs(Register reg, std::vector<LiveInterval*>* generated);
  Register getAllowedAlias(LiveInterval* interv, int phyidx);
  void seedLiveRegs();
  bool hasVirtRegAlloc();
  void splitEdges();
  void splitByLoop();
  void splitByFcall();
  void splitIntervals();
  void dumpInfo();
  void assignReg(std::vector<int>& result);
  void dumpIntervals();
  void dumpEdges();
  void finalizeAlloc();
  float getPointFreq(SlotIndex index);
  void specifyEdges();
  void makeDConfig(std::vector<int>& result);
  void loadConfig(std::vector<int>& result);

  bool isCoalCopy(const MachineInstr* mi);
  bool instWithHints(MachineInstr& mi);
  void set_ana(MachineFunction &mf);
  void get_ana(MachineFunction &mf);
  

  bool isACalleeSavedRegister(MCRegister Reg);
  void updateIntervals(SmallVector<Register, 4>& newvregs, std::list<LiveInterval*>& newintervs);
  void makeNoRematInsts(LiveInterval* interv, std::set<MachineInstr*>& insts, std::set<LiveInterval*>& spilled);
  void updateRegMap();

  int getFromRegID(int reg_id) {
    for (int i = 0; i < int(to_ids.size()); i++) {
      if (to_ids[i] == reg_id) {
        return from_ids[i];
      }
    }
    return -1;
  }
  int getToRegID(int reg_id) {
    for (int i = 0; i < int(from_ids.size()); i++) {
      if (from_ids[i] == reg_id) {
        return to_ids[i];
      }
    }
    return -1;
  }
};
} // namespace llvm
#endif // #ifndef LLVM_CODEGEN_REGALLOCGRAD_H_
