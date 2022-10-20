//===-- IRMutator.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/FuzzMutate/IRMutator.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/FuzzMutate/OpDescriptor.h"
#include "llvm/FuzzMutate/Operations.h"
#include "llvm/FuzzMutate/Random.h"
#include "llvm/FuzzMutate/RandomIRBuilder.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

void IRMutationStrategy::mutate(Module &M, RandomIRBuilder &IB) {
  auto RS = makeSampler<Function *>(IB.Rand);
  for (Function &F : M)
    if (!F.isDeclaration())
      RS.sample(&F, /*Weight=*/1);

  // Should we have more functions?
  while (RS.totalWeight() < 1) {
    Function *F = IB.createFunctionDefinition(M, 0, 5);
    RS.sample(F, /*Weight=*/1);
  }
  mutate(*RS.getSelection(), IB);
}

void IRMutationStrategy::mutate(Function &F, RandomIRBuilder &IB) {
  mutate(*makeSampler(IB.Rand, make_pointer_range(F)).getSelection(), IB);
}

void IRMutationStrategy::mutate(BasicBlock &BB, RandomIRBuilder &IB) {
  mutate(*makeSampler(IB.Rand, make_pointer_range(BB)).getSelection(), IB);
}

void IRMutator::mutateModule(Module &M, int Seed, uint64_t CurSize,
                             uint64_t MaxSize) {
  std::vector<Type *> Types;
  for (const auto &Getter : AllowedTypes)
    Types.push_back(Getter(M.getContext()));
  RandomIRBuilder IB(Seed, Types);

  auto RS = makeSampler<IRMutationStrategy *>(IB.Rand);
  for (const auto &Strategy : Strategies)
    RS.sample(Strategy.get(),
              Strategy->getWeight(CurSize, MaxSize, RS.totalWeight()));
  auto Strategy = RS.getSelection();

  Strategy->mutate(M, IB);
}

static void eliminateDeadCode(Function &F) {
  FunctionPassManager FPM;
  FPM.addPass(DCEPass());
  FunctionAnalysisManager FAM;
  FAM.registerPass([&] { return TargetLibraryAnalysis(); });
  FAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  FPM.run(F, FAM);
}

void InjectorIRStrategy::mutate(Function &F, RandomIRBuilder &IB) {
  IRMutationStrategy::mutate(F, IB);
  eliminateDeadCode(F);
}

std::vector<fuzzerop::OpDescriptor> InjectorIRStrategy::getDefaultOps() {
  std::vector<fuzzerop::OpDescriptor> Ops;
  describeFuzzerIntOps(Ops);
  describeFuzzerFloatOps(Ops);
  describeFuzzerControlFlowOps(Ops);
  describeFuzzerPointerOps(Ops);
  describeFuzzerAggregateOps(Ops);
  describeFuzzerVectorOps(Ops);
  describeFuzzerUnaryOperations(Ops);
  describeFuzzerOtherOps(Ops);
  return Ops;
}

Optional<fuzzerop::OpDescriptor>
InjectorIRStrategy::chooseOperation(Value *Src, RandomIRBuilder &IB) {
  auto OpMatchesPred = [&Src](fuzzerop::OpDescriptor &Op) {
    return Op.SourcePreds[0].matches({}, Src);
  };
  auto RS = makeSampler(IB.Rand, make_filter_range(Operations, OpMatchesPred));
  if (RS.isEmpty())
    return None;
  return *RS;
}

void InjectorIRStrategy::mutate(BasicBlock &BB, RandomIRBuilder &IB) {
  SmallVector<Instruction *, 32> Insts;
  for (auto I = BB.getFirstInsertionPt(), E = BB.end(); I != E; ++I)
    Insts.push_back(&*I);
  if (Insts.size() < 1)
    return;

  // Choose an insertion point for our new instruction.
  uint64_t IP = uniform<uint64_t>(IB.Rand, 0, Insts.size() - 1);

  auto InstsBefore = makeArrayRef(Insts).slice(0, IP);
  auto InstsAfter = makeArrayRef(Insts).slice(IP);

  // Choose a source, which will be used to constrain the operation selection.
  SmallVector<Value *, 2> Srcs;
  Srcs.push_back(IB.findOrCreateSource(BB, InstsBefore));

  // Choose an operation that's constrained to be valid for the type of the
  // source, collect any other sources it needs, and then build it.
  auto OpDesc = chooseOperation(Srcs[0], IB);
  // Bail if no operation was found
  if (!OpDesc)
    return;

  for (const auto &Pred : makeArrayRef(OpDesc->SourcePreds).slice(Srcs.size()))
    Srcs.push_back(IB.findOrCreateSource(BB, InstsBefore, Srcs, Pred));

  if (Value *Op = OpDesc->BuilderFunc(Srcs, Insts[IP])) {
    // Find a sink and wire up the results of the operation.
    IB.connectToSink(BB, InstsAfter, Op);
  }
}

uint64_t InstDeleterIRStrategy::getWeight(uint64_t CurrentSize,
                                          uint64_t MaxSize,
                                          uint64_t CurrentWeight) {
  // If we have less than 200 bytes, panic and try to always delete.
  if (CurrentSize > MaxSize - 200)
    return CurrentWeight ? CurrentWeight * 100 : 1;
  // Draw a line starting from when we only have 1k left and increasing linearly
  // to double the current weight.
  int64_t Line = (-2 * static_cast<int64_t>(CurrentWeight)) *
                 (static_cast<int64_t>(MaxSize) -
                  static_cast<int64_t>(CurrentSize) - 1000) /
                 1000;
  // Clamp negative weights to zero.
  if (Line < 0)
    return 0;
  return Line;
}

void InstDeleterIRStrategy::mutate(Function &F, RandomIRBuilder &IB) {
  auto RS = makeSampler<Instruction *>(IB.Rand);
  for (Instruction &Inst : instructions(F)) {
    // TODO: We can't handle these instructions.
    if (Inst.isTerminator() || Inst.isEHPad() || Inst.isSwiftError() ||
        isa<PHINode>(Inst))
      continue;

    RS.sample(&Inst, /*Weight=*/1);
  }
  if (RS.isEmpty())
    return;

  // Delete the instruction.
  mutate(*RS.getSelection(), IB);
  // Clean up any dead code that's left over after removing the instruction.
  eliminateDeadCode(F);
}

void InstDeleterIRStrategy::mutate(Instruction &Inst, RandomIRBuilder &IB) {
  assert(!Inst.isTerminator() && "Deleting terminators invalidates CFG");

  if (Inst.getType()->isVoidTy()) {
    // Instructions with void type (ie, store) have no uses to worry about. Just
    // erase it and move on.
    Inst.eraseFromParent();
    return;
  }

  // Otherwise we need to find some other value with the right type to keep the
  // users happy.
  auto Pred = fuzzerop::onlyType(Inst.getType());
  auto RS = makeSampler<Value *>(IB.Rand);
  SmallVector<Instruction *, 32> InstsBefore;
  BasicBlock *BB = Inst.getParent();
  for (auto I = BB->getFirstInsertionPt(), E = Inst.getIterator(); I != E;
       ++I) {
    if (Pred.matches({}, &*I))
      RS.sample(&*I, /*Weight=*/1);
    InstsBefore.push_back(&*I);
  }
  if (!RS)
    RS.sample(IB.newSource(*BB, InstsBefore, {}, Pred), /*Weight=*/1);

  Inst.replaceAllUsesWith(RS.getSelection());
  Inst.eraseFromParent();
}

void InstModificationIRStrategy::mutate(Instruction &Inst,
                                        RandomIRBuilder &IB) {
  SmallVector<std::function<void()>, 8> Modifications;
  CmpInst *CI = nullptr;
  GetElementPtrInst *GEP = nullptr;
  switch (Inst.getOpcode()) {
  default:
    break;
  case Instruction::Add:
  case Instruction::Mul:
  case Instruction::Sub:
  case Instruction::Shl:
    Modifications.push_back([&Inst]() { Inst.setHasNoSignedWrap(true); });
    Modifications.push_back([&Inst]() { Inst.setHasNoSignedWrap(false); });
    Modifications.push_back([&Inst]() { Inst.setHasNoUnsignedWrap(true); });
    Modifications.push_back([&Inst]() { Inst.setHasNoUnsignedWrap(false); });

    break;
  case Instruction::ICmp:
    CI = cast<ICmpInst>(&Inst);
    Modifications.push_back([CI]() { CI->setPredicate(CmpInst::ICMP_EQ); });
    Modifications.push_back([CI]() { CI->setPredicate(CmpInst::ICMP_NE); });
    Modifications.push_back([CI]() { CI->setPredicate(CmpInst::ICMP_UGT); });
    Modifications.push_back([CI]() { CI->setPredicate(CmpInst::ICMP_UGE); });
    Modifications.push_back([CI]() { CI->setPredicate(CmpInst::ICMP_ULT); });
    Modifications.push_back([CI]() { CI->setPredicate(CmpInst::ICMP_ULE); });
    Modifications.push_back([CI]() { CI->setPredicate(CmpInst::ICMP_SGT); });
    Modifications.push_back([CI]() { CI->setPredicate(CmpInst::ICMP_SGE); });
    Modifications.push_back([CI]() { CI->setPredicate(CmpInst::ICMP_SLT); });
    Modifications.push_back([CI]() { CI->setPredicate(CmpInst::ICMP_SLE); });
    break;
  case Instruction::GetElementPtr:
    GEP = cast<GetElementPtrInst>(&Inst);
    Modifications.push_back([GEP]() { GEP->setIsInBounds(true); });
    Modifications.push_back([GEP]() { GEP->setIsInBounds(false); });
    break;
  }

  auto RS = makeSampler(IB.Rand, Modifications);
  if (RS)
    RS.getSelection()();
}

void FunctionIRStrategy::mutate(BasicBlock &BB, RandomIRBuilder &IB) {
  Module *M = BB.getParent()->getParent();
  // If nullptr is selected, we will create a new function declaration.
  SmallVector<Function *, 32> Functions({nullptr});
  for (Function &F : M->functions()) {
    Functions.push_back(&F);
  }

  auto RS = makeSampler(IB.Rand, Functions);
  Function *F = RS.getSelection();
  if (!F) {
    F = IB.createFunctionDeclaration(*M, 0, 5);
  }

  FunctionType *FTy = F->getFunctionType();
  SmallVector<fuzzerop::SourcePred, 2> SourcePreds;
  if (F->arg_size() != 0) {
    for (Type *ArgTy : FTy->params()) {
      SourcePreds.push_back(fuzzerop::SourcePred(ArgTy));
    }
  }
  bool isRetVoid = (F->getReturnType() == Type::getVoidTy(M->getContext()));
  auto BuilderFunc = [FTy, F, isRetVoid](ArrayRef<Value *> Srcs,
                                         Instruction *Inst) {
    StringRef Name = isRetVoid ? nullptr : "C";
    CallInst *Call = CallInst::Create(FTy, F, Srcs, Name, Inst);
    // Don't return this call inst if it return void as it can't be sinked.
    return isRetVoid ? nullptr : Call;
  };

  SmallVector<Instruction *, 32> Insts;
  for (auto I = BB.getFirstInsertionPt(), E = BB.end(); I != E; ++I)
    Insts.push_back(&*I);
  if (Insts.size() < 1)
    return;

  // Choose an insertion point for our new call instruction.
  uint64_t IP = uniform<uint64_t>(IB.Rand, 0, Insts.size() - 1);

  auto InstsBefore = makeArrayRef(Insts).slice(0, IP);
  auto InstsAfter = makeArrayRef(Insts).slice(IP);

  // Choose a source, which will be used to constrain the operation selection.
  SmallVector<Value *, 2> Srcs;

  for (const auto &Pred : makeArrayRef(SourcePreds)) {
    Srcs.push_back(IB.findOrCreateSource(BB, InstsBefore, Srcs, Pred));
  }

  if (Value *Op = BuilderFunc(Srcs, Insts[IP])) {
    // Find a sink and wire up the results of the operation.
    IB.connectToSink(BB, InstsAfter, Op);
  }
}

void CFGIRStrategy::mutate(BasicBlock &BB, RandomIRBuilder &IB) {
  SmallVector<Instruction *, 32> Insts;
  for (auto I = BB.getFirstInsertionPt(), E = BB.end(); I != E; ++I)
    Insts.push_back(&*I);
  if (Insts.size() < 1)
    return;

  // Choose an insertion point for our new call instruction.
  uint64_t IP = uniform<uint64_t>(IB.Rand, 0, Insts.size() - 1);
  auto InstsBefore = makeArrayRef(Insts).slice(0, IP);

  // Next inherits Blocks' terminator.
  // Here, we have to create a new terminator for Block.
  BasicBlock *Block = Insts[IP]->getParent();
  BasicBlock *Next = Block->splitBasicBlock(Insts[IP], "BB");

  Function *F = BB.getParent();
  LLVMContext &C = F->getParent()->getContext();
  bool coin = uniform<uint64_t>(IB.Rand, 0, 1);
  // A coin decides if it is branch or switch
  if (coin) {
    BasicBlock *IfTrue = BasicBlock::Create(C, "BR_T", F);
    BasicBlock *IfFalse = BasicBlock::Create(C, "BR_F", F);
    Value *Cond =
        IB.findOrCreateSource(*Block, InstsBefore, {},
                              fuzzerop::SourcePred(Type::getInt1Ty(C)), false);
    BranchInst *Branch = BranchInst::Create(IfTrue, IfFalse, Cond);
    // Remove the old terminator.
    ReplaceInstWithInst(Block->getTerminator(), Branch);
    // Use these blocks
    connectBlocksToSink({IfTrue, IfFalse}, Next, IB);
  } else {
    // Determine Integer type.
    auto isIntegerType = [](Type *Ty) { return Ty->isIntegerTy(); };
    auto RS =
        makeSampler(IB.Rand, make_filter_range(IB.KnownTypes, isIntegerType));
    if (!RS) {
      llvm_unreachable("Shouldn't be here");
    }
    IntegerType *Ty = dyn_cast<IntegerType>(RS.getSelection());
    if (!Ty) {
      llvm_unreachable("Shouldn't be here");
    }
    uint64_t BitSize = Ty->getBitWidth();
    uint64_t MaxCaseVal =
        BitSize >= 64 ? 0xffffffffffffffff : ((uint64_t)1 << BitSize) - 1;

    // Create Switch inst in Block
    Value *Cond = IB.findOrCreateSource(*Block, InstsBefore, {},
                                        fuzzerop::SourcePred(Ty), false);
    BasicBlock *DefaultBlock = BasicBlock::Create(C, "SW_D", F);
    uint64_t NumCase = uniform<uint64_t>(IB.Rand, 3, 8);
    // Make sure we don't have more case than this type can handle.
    if (NumCase > (1 << BitSize)) {
      NumCase = 1 << BitSize;
    }
    SwitchInst *Switch = SwitchInst::Create(Cond, DefaultBlock, NumCase);
    // Remove the old terminator.
    ReplaceInstWithInst(Block->getTerminator(), Switch);

    // Create blocks, for each block assign a case value.
    SmallVector<BasicBlock *, 4> Blocks({DefaultBlock});
    SmallSet<uint64_t, 4> CaseVals;
    for (uint64_t i = 0; i < NumCase; i++) {
      uint64_t CaseVal = [&CaseVals, MaxCaseVal, &IB]() {
        uint64_t tmp;
        // Make sure we don't have two cases with same value.
        do {
          tmp = uniform<uint64_t>(IB.Rand, 0, MaxCaseVal);
        } while (CaseVals.count(tmp) != 0);
        CaseVals.insert(tmp);
        return tmp;
      }();

      BasicBlock *CaseBlock = BasicBlock::Create(C, "SW_C", F);
      ConstantInt *OnValue = ConstantInt::get(Ty, CaseVal);
      Switch->addCase(OnValue, CaseBlock);
      Blocks.push_back(CaseBlock);
    }

    // Use these blocks
    connectBlocksToSink(Blocks, Next, IB);
  }
}

enum CFGToSink {
  Return,
  DirectSink,
  // SinkOrPeer,
  SinkOrSelfLoop,
  EndOfCFGToLink
};

void CFGIRStrategy::connectBlocksToSink(ArrayRef<BasicBlock *> Blocks,
                                        BasicBlock *Sink, RandomIRBuilder &IB) {
  uint64_t Idx = uniform<uint64_t>(IB.Rand, 0, Blocks.size() - 1);
  for (uint64_t I = 0; I < Blocks.size(); I++) {
    CFGToSink ToSink = static_cast<CFGToSink>(
        uniform<uint64_t>(IB.Rand, 0, CFGToSink::EndOfCFGToLink - 1));
    // We have at least one block that directly goes to sink.
    if (I == Idx) {
      ToSink = CFGToSink::DirectSink;
    }
    BasicBlock *BB = Blocks[I];
    Function *F = BB->getParent();
    LLVMContext &C = F->getParent()->getContext();
    switch (ToSink) {
    case CFGToSink::Return: {
      Type *RetTy = F->getReturnType();
      Value *RetValue = nullptr;
      if (RetTy != Type::getVoidTy(C)) {
        RetValue =
            IB.findOrCreateSource(*BB, {}, {}, fuzzerop::SourcePred(RetTy));
      }
      ReturnInst::Create(C, RetValue, BB);
      break;
    }
    case CFGToSink::DirectSink: {
      BranchInst::Create(Sink, BB);
      break;
    }
    case CFGToSink::SinkOrSelfLoop: {
      SmallVector<BasicBlock *, 2> Branches({Sink, BB});
      uint64_t coin = uniform<uint64_t>(IB.Rand, 0, 1);
      Value *Cond = IB.findOrCreateSource(
          *BB, {}, {}, fuzzerop::SourcePred(Type::getInt1Ty(C)), false);
      BranchInst::Create(Branches[coin], Branches[1 - coin], Cond, BB);
      break;
    }
    case CFGToSink::EndOfCFGToLink:
      llvm_unreachable("Shouldn't be here");
    }
  }
}

void InsertPHItrategy::mutate(BasicBlock &BB, RandomIRBuilder &IB) {
  // Can't insert PHI node to entry node.
  if (&BB == &BB.getParent()->getEntryBlock()) {
    return;
  }
  Type *Ty = IB.randomType();
  PHINode *PHI = PHINode::Create(Ty, llvm::pred_size(&BB), "", &BB.front());
  auto Pred = fuzzerop::SourcePred(Ty);

  SmallVector<Value *, 4> Srcs;
  for (BasicBlock *Prev : predecessors(&BB)) {
    SmallVector<Instruction *, 32> Insts;
    for (auto I = Prev->begin(); I != Prev->end(); ++I)
      Insts.push_back(&*I);
    Value *Src = IB.findOrCreateSource(*Prev, Insts, Srcs, Pred);
    Srcs.push_back(Src);
    PHI->addIncoming(Src, Prev);
  }
  SmallVector<Instruction *, 32> InstsAfter;
  for (auto I = BB.getFirstInsertionPt(), E = BB.end(); I != E; ++I)
    InstsAfter.push_back(&*I);
  IB.connectToSink(BB, InstsAfter, PHI);
}

void OperandMutatorstrategy::mutate(Function &F, RandomIRBuilder &IB) {
  for (BasicBlock &BB : F) {
    this->mutate(BB, IB);
  }
}
void OperandMutatorstrategy::mutate(BasicBlock &BB, RandomIRBuilder &IB) {
  SmallVector<Instruction *, 32> Insts;
  for (auto I = BB.getFirstInsertionPt(), E = BB.end(); I != E; ++I)
    Insts.push_back(&*I);
  if (Insts.size() < 1)
    return;

  // Choose an insertion point for our new instruction.
  uint64_t IP = uniform<uint64_t>(IB.Rand, 0, Insts.size() - 1);

  auto InstsBefore = makeArrayRef(Insts).slice(0, IP);
  Instruction *Inst = Insts[IP];
  uint64_t NumOperands = Inst->getNumOperands();
  uint64_t Idx = uniform<uint64_t>(IB.Rand, 0, NumOperands - 1);
  Type *Ty = Inst->getOperand(Idx)->getType();
  // Changing these types may potentially break the module.
  if (Ty->isLabelTy() || Ty->isMetadataTy() || Ty->isFunctionTy() ||
      Ty->isPointerTy()) {
    return;
  }
  Value *NewOperand =
      IB.findOrCreateSource(BB, InstsBefore, {}, fuzzerop::SourcePred(Ty));
  Inst->setOperand(Idx, NewOperand);
}

std::unique_ptr<Module> llvm::parseModule(const uint8_t *Data, uint64_t Size,
                                          LLVMContext &Context) {

  if (Size <= 1)
    // We get bogus data given an empty corpus - just create a new module.
    return std::make_unique<Module>("M", Context);

  auto Buffer = MemoryBuffer::getMemBuffer(
      StringRef(reinterpret_cast<const char *>(Data), Size), "Fuzzer input",
      /*RequiresNullTerminator=*/false);

  SMDiagnostic Err;
  auto M = parseBitcodeFile(Buffer->getMemBufferRef(), Context);
  if (Error E = M.takeError()) {
    errs() << toString(std::move(E)) << "\n";
    return nullptr;
  }
  return std::move(M.get());
}

uint64_t llvm::writeModule(const Module &M, uint8_t *Dest, uint64_t MaxSize) {
  std::string Buf;
  {
    raw_string_ostream OS(Buf);
    WriteBitcodeToFile(M, OS);
  }
  if (Buf.size() > MaxSize)
    return 0;
  memcpy(Dest, Buf.data(), Buf.size());
  return Buf.size();
}

std::unique_ptr<Module> llvm::parseAndVerify(const uint8_t *Data, uint64_t Size,
                                             LLVMContext &Context) {
  auto M = parseModule(Data, Size, Context);
  if (!M || verifyModule(*M, &errs()))
    return nullptr;

  return M;
}
