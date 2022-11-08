//===-- RandomIRBuilder.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/FuzzMutate/RandomIRBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/FuzzMutate/OpDescriptor.h"
#include "llvm/FuzzMutate/Random.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"

using namespace llvm;
using namespace fuzzerop;

/// Return a vector of Blocks that dominates this block, excluding current
/// block.
template <typename DomTree = DominatorTree>
std::vector<BasicBlock *> getDominators(BasicBlock *BB) {
  std::vector<BasicBlock *> ret;
  DomTree DT(*BB->getParent());
  DomTreeNode *Node = DT[BB]->getIDom();
  while (Node && Node->getBlock()) {
    ret.push_back(Node->getBlock());
    // Get parent block.
    Node = Node->getIDom();
  }
  return ret;
}

/// Return a vector of Blocks that is dominated by this block, excluding current
/// block
template <typename DomTree = DominatorTree>
std::vector<BasicBlock *> getDominees(BasicBlock *BB) {
  DomTree DT(*BB->getParent());
  std::vector<BasicBlock *> ret({BB});
  uint64_t Idx = 0;
  while (Idx < ret.size()) {
    DomTreeNode *Node = DT[ret[Idx]];
    Idx++;
    for (DomTreeNode *Child : Node->children()) {
      ret.push_back(Child->getBlock());
    }
  }
  // We don't add first Node, as it is ourself.
  ret.erase(ret.begin());
  return ret;
}

/// Generatie a random permutation from `[0..Max)`.
SmallVector<uint64_t, 4> RandomIRBuilder::getRandomPermutation(uint64_t Max) {
  SmallVector<uint64_t, 4> Ret;
  for (uint64_t i = 0; i < Max; i++) {
    Ret.push_back(i);
  }
  // Get a random permutation by swapping.
  // Use int64_t so that when Max == 0, this loop doesn't execute.
  for (int64_t i = Max - 1; i > 0; i--) {
    std::swap(Ret[i], Ret[Rand() % i]);
  }
  return Ret;
}

GlobalVariable *
RandomIRBuilder::findOrCreateGlobalVariable(Module *M, ArrayRef<Value *> Srcs,
                                            fuzzerop::SourcePred Pred,
                                            bool *DidCreate) {
  auto MatchesPred = [&Srcs, &Pred](GlobalVariable *GV) {
    // Can't directly compare GV's type, as it would be a pointer to the actual
    // type.
    return Pred.matches(Srcs, GV->getInitializer());
  };
  SmallVector<GlobalVariable *, 4> GlobalVars;
  for (GlobalVariable &GV : M->globals()) {
    GlobalVars.push_back(&GV);
  }
  auto RS = makeSampler(Rand, make_filter_range(GlobalVars, MatchesPred));
  RS.sample(nullptr, 1);
  GlobalVariable *GV = RS.getSelection();
  if (!GV) {
    if (DidCreate)
      *DidCreate = true;
    using LinkageTypes = GlobalVariable::LinkageTypes;
    auto TRS = makeSampler<Constant *>(Rand);
    /// FIXME: This might be incorrect for operands that needs to be constant.
    /// We shouldn't generate a constnat and save it.
    TRS.sample(Pred.generate(Srcs, KnownTypes));
    Constant *Init = TRS.getSelection();
    Type *Ty = Init->getType();
    GV = new GlobalVariable(*M, Ty, false, LinkageTypes::ExternalLinkage, Init,
                            "G");
  }
  return GV;
}

enum SourceType {
  PrevValueInBlock,
  FunctionArgument,
  ValueInDominator,
  SrcFromGlobalVar,
  NewSource,
  EndOfValueSource,
};
Value *RandomIRBuilder::findOrCreateSource(BasicBlock &BB,
                                           ArrayRef<Instruction *> Insts) {
  return findOrCreateSource(BB, Insts, {}, anyType());
}

/// @brief Given the Sources that already satisfied Pred, find next source.
/// @param BB Current basic block.
/// @param Insts Instructions in this basic block before insertion point
/// @param Srcs Sources that have been selected
/// @param Pred The predicate that the sources needs to satisfy
/// @return The next source Value.
Value *RandomIRBuilder::findOrCreateSource(BasicBlock &BB,
                                           ArrayRef<Instruction *> Insts,
                                           ArrayRef<Value *> Srcs,
                                           SourcePred Pred,
                                           bool AllowConstant) {
  auto MatchesPred = [&Srcs, &Pred](Value *V) { return Pred.matches(Srcs, V); };
  auto SrcTys = getRandomPermutation(EndOfValueSource);
  for (uint64_t SrcTy : SrcTys) {
    switch (SrcTy) {
    case PrevValueInBlock: {
      auto RS = makeSampler(Rand, make_filter_range(Insts, MatchesPred));
      if (!RS.isEmpty()) {
        return RS.getSelection();
      }
      break;
    }
    case FunctionArgument: {
      Function *F = BB.getParent();
      // Somehow I can't use iterators to init these vectors, it will have type
      // mismatch.
      std::vector<Argument *> Args;
      for (uint64_t i = 0; i < F->arg_size(); i++) {
        Args.push_back(F->getArg(i));
      }
      auto RS = makeSampler(Rand, make_filter_range(Args, MatchesPred));
      if (!RS.isEmpty()) {
        return RS.getSelection();
      }
      break;
    }
    case ValueInDominator: {
      auto Dominators = getDominators(&BB);
      auto P = getRandomPermutation(Dominators.size());
      for (uint64_t i : P) {
        BasicBlock *Dom = Dominators[i];
        // Somehow I can't use iterators to init these vectors, it will have
        // type mismatch.
        std::vector<Instruction *> Instructions;
        for (Instruction &I : *Dom) {
          Instructions.push_back(&I);
        }
        auto RS =
            makeSampler(Rand, make_filter_range(Instructions, MatchesPred));
        // Also consider choosing no source, meaning we want a new one.
        if (!RS.isEmpty()) {
          return RS.getSelection();
        }
      }
      break;
    }
    case SrcFromGlobalVar: {
      Module *M = BB.getParent()->getParent();
      bool DidCreate = false;
      GlobalVariable *GV =
          findOrCreateGlobalVariable(M, Srcs, Pred, &DidCreate);
      Type *Ty = GV->getType()->getNonOpaquePointerElementType();
      LoadInst *LoadGV = nullptr;
      if (BB.getTerminator()) {
        LoadGV = new LoadInst(Ty, GV, "LGV", &*BB.getFirstInsertionPt());
      } else {
        LoadGV = new LoadInst(Ty, GV, "LGV", &BB);
      }
      // Because we might be generating new values, we have to check if it
      // matches again.
      if (DidCreate) {
        if (Pred.matches(Srcs, LoadGV)) {
          return LoadGV;
        } else {
          LoadGV->eraseFromParent();
          // If no one is using this GlobalVariable, delete it too.
          if (GV->hasNUses(0)) {
            GV->eraseFromParent();
          }
        }
      }
      break;
    }
    case NewSource: {
      return newSource(BB, Insts, Srcs, Pred, AllowConstant);
    }
    case EndOfValueSource: {
      llvm_unreachable("Should've found a new source type.");
    }
    }
  }
  llvm_unreachable("Should've found a new source.");
}

Value *RandomIRBuilder::newSource(BasicBlock &BB, ArrayRef<Instruction *> Insts,
                                  ArrayRef<Value *> Srcs, SourcePred Pred,
                                  bool AllowConstant) {
  // Generate some constants to choose from.
  auto RS = makeSampler<Value *>(Rand);
  if (AllowConstant) {
    RS.sample(Pred.generate(Srcs, KnownTypes));
  }

  // If we can find a pointer to load from, use it half the time.
  Value *Ptr = findPointer(BB, Insts, Srcs, Pred);
  if (Ptr) {
    // Create load from the chosen pointer
    auto IP = BB.getFirstInsertionPt();
    if (auto *I = dyn_cast<Instruction>(Ptr)) {
      IP = ++I->getIterator();
      assert(IP != BB.end() && "guaranteed by the findPointer");
    }
    // For opaque pointers, pick the type independently.
    Type *AccessTy = Ptr->getType()->isOpaquePointerTy()
                         ? RS.getSelection()->getType()
                         : Ptr->getType()->getNonOpaquePointerElementType();
    auto *NewLoad = new LoadInst(AccessTy, Ptr, "L", &*IP);

    // Only sample this load if it really matches the descriptor
    if (Pred.matches(Srcs, NewLoad))
      RS.sample(NewLoad, RS.totalWeight());
    else
      NewLoad->eraseFromParent();
  }

  // We don't have anything at this point, create a stack memory.
  if (RS.isEmpty()) {
    // Randomly select a type that is allowed here.
    auto TRS = makeSampler<Value *>(Rand);
    TRS.sample(Pred.generate(Srcs, KnownTypes));
    Value *V = TRS.getSelection();
    Type *Ty = V->getType();
    // Generate a stack alloca.
    Function *F = BB.getParent();
    BasicBlock *EntryBB = &F->getEntryBlock();
    /// TODO: For all Allocas, maybe allocate an array.
    AllocaInst *Alloca = new AllocaInst(Ty, 0, "A", EntryBB->getTerminator());
    LoadInst *Load = nullptr;
    // nstantInt *ArrLen =
    //     ConstantInt::get(IntegerType::get(F->getParent()->getContext(), 32),
    //     1);
    if (BB.getTerminator()) {
      Load = new LoadInst(Ty, Alloca, /*ArrLen,*/ "L", BB.getTerminator());
    } else {
      Load = new LoadInst(Ty, Alloca, /*ArrLen,*/ "L", &BB);
    }
    RS.sample(Load, 1);
  }

  assert(!RS.isEmpty() && "Failed to generate sources");
  return RS.getSelection();
}

static bool isCompatibleReplacement(const Instruction *I, const Use &Operand,
                                    const Value *Replacement) {
  if (Operand->getType() != Replacement->getType())
    return false;
  unsigned int OperandNo = Operand.getOperandNo();
  switch (I->getOpcode()) {
  case Instruction::GetElementPtr:
  case Instruction::ExtractElement:
  case Instruction::ExtractValue:
    // TODO: We could potentially validate these, but for now just leave
    // indices alone.
    if (OperandNo >= 1)
      return false;
    break;
  case Instruction::InsertValue:
  case Instruction::InsertElement:
  case Instruction::ShuffleVector:
    if (OperandNo >= 2)
      return false;
    break;
  // For Br/Switch, we only try to modify the 1st Operand(Cond).
  // Modify other operands, like switch case may accidently change case from
  // ConstnatInt to a register, which is illegal.
  case Instruction::Switch:
  case Instruction::Br:
    if (OperandNo >= 1)
      return false;
    break;
  default:
    break;
  }
  return true;
}

enum SinkType {
  NextValueInBlock,
  PointersInDom,
  InstInDominees,
  NewSink,
  SinkToGlobalVar,
  EndOfValueSink,
};
void RandomIRBuilder::connectToSink(BasicBlock &BB,
                                    ArrayRef<Instruction *> Insts, Value *V) {
  auto SinkTys = getRandomPermutation(EndOfValueSink);
  auto findSinkAndConnect = [this, V](ArrayRef<Instruction *> Instructions) {
    auto RS = makeSampler<Use *>(Rand);
    for (auto &I : Instructions) {
      if (isa<IntrinsicInst>(I))
        // TODO: Replacing operands of intrinsics would be interesting,
        // but there's no easy way to verify that a given replacement is
        // valid given that intrinsics can impose arbitrary constraints.
        continue;
      for (Use &U : I->operands())
        if (isCompatibleReplacement(I, U, V))
          RS.sample(&U, 1);
    }
    if (!RS.isEmpty()) {
      Use *Sink = RS.getSelection();
      User *U = Sink->getUser();
      unsigned OpNo = Sink->getOperandNo();
      U->setOperand(OpNo, V);
      return true;
    }
    return false;
  };
  for (uint64_t SinkTy : SinkTys) {
    switch (SinkTy) {
    case NextValueInBlock:
      if (findSinkAndConnect(Insts))
        return;
      break;
    case PointersInDom: {
      auto Dominators = getDominators(&BB);
      for (BasicBlock *Dom : Dominators) {
        for (Instruction &I : *Dom) {
          if (PointerType *PtrTy = dyn_cast<PointerType>(I.getType())) {
            if (PtrTy->isOpaqueOrPointeeTypeMatches(V->getType())) {
              new StoreInst(V, &I, Insts.back());
              return;
            }
          }
        }
      }
      /// TODO: Also consider pointers in function argument.
      break;
    }
    case InstInDominees: {
      auto Dominees = getDominees(&BB);
      auto permutation = getRandomPermutation(Dominees.size());
      for (uint64_t i : permutation) {
        BasicBlock *Dominee = Dominees[i];
        // Somehow I can't use iterators to init these vectors, it will have
        // type mismatch.
        std::vector<Instruction *> Instructions;
        for (Instruction &I : *Dominee)
          Instructions.push_back(&I);
        if (findSinkAndConnect(Instructions))
          return;
      }
      break;
    }
    case SinkToGlobalVar: {
      Module *M = BB.getParent()->getParent();
      GlobalVariable *GV =
          findOrCreateGlobalVariable(M, {}, fuzzerop::onlyType(V->getType()));
      new StoreInst(V, GV, Insts.back());
      break;
    }
    case NewSink:
      newSink(BB, Insts, V);
      return;
    case EndOfValueSink:
    default:
      llvm_unreachable("Should've found a new sink type.");
    };
  }
}

void RandomIRBuilder::newSink(BasicBlock &BB, ArrayRef<Instruction *> Insts,
                              Value *V) {
  Value *Ptr = findPointer(BB, Insts, {V}, matchFirstType());
  if (!Ptr) {
    if (uniform(Rand, 0, 1))
      Ptr = new AllocaInst(V->getType(), 0, "A", &*BB.getFirstInsertionPt());
    else
      Ptr = UndefValue::get(PointerType::get(V->getType(), 0));
  }

  new StoreInst(V, Ptr, Insts.back());
}

Value *RandomIRBuilder::findPointer(BasicBlock &BB,
                                    ArrayRef<Instruction *> Insts,
                                    ArrayRef<Value *> Srcs, SourcePred Pred) {
  auto IsMatchingPtr = [&Srcs, &Pred](Instruction *Inst) {
    // Invoke instructions sometimes produce valid pointers but
    // currently we can't insert loads or stores from them
    if (Inst->isTerminator())
      return false;

    if (auto *PtrTy = dyn_cast<PointerType>(Inst->getType())) {
      if (PtrTy->isOpaque())
        return true;

      // We can never generate loads from non first class or non sized types
      Type *ElemTy = PtrTy->getNonOpaquePointerElementType();
      if (!ElemTy->isSized() || !ElemTy->isFirstClassType())
        return false;

      // TODO: Check if this is horribly expensive.
      return Pred.matches(Srcs, UndefValue::get(ElemTy));
    }
    return false;
  };
  if (auto RS = makeSampler(Rand, make_filter_range(Insts, IsMatchingPtr)))
    return RS.getSelection();
  return nullptr;
}

Type *RandomIRBuilder::randomType() {
  uint64_t TyIdx = uniform<uint64_t>(Rand, 0, KnownTypes.size() - 1);
  return KnownTypes[TyIdx];
}

Function *RandomIRBuilder::createFunctionDeclaration(Module &M,
                                                     uint64_t ArgNum) {
  Type *RetType = randomType();

  SmallVector<Type *, 2> Args;
  for (uint64_t i = 0; i < ArgNum; i++) {
    Args.push_back(randomType());
  }

  Function *F = Function::Create(FunctionType::get(RetType, Args,
                                                   /*isVarArg=*/false),
                                 GlobalValue::ExternalLinkage, "f", &M);
  return F;
}
Function *RandomIRBuilder::createFunctionDeclaration(Module &M,
                                                     uint64_t MinArgNum,
                                                     uint64_t MaxArgNum) {
  return createFunctionDeclaration(
      M, uniform<uint64_t>(Rand, MinArgNum, MaxArgNum));
}

Function *RandomIRBuilder::createFunctionDefinition(Module &M,
                                                    uint64_t ArgNum) {
  Function *F = this->createFunctionDeclaration(M, ArgNum);

  // TODO: Some arguments and a return value would probably be more
  // interesting.
  LLVMContext &Context = M.getContext();
  BasicBlock *BB = BasicBlock::Create(Context, "BB", F);
  Type *RetTy = F->getReturnType();
  if (RetTy != Type::getVoidTy(Context)) {
    Instruction *RetAlloca = new AllocaInst(RetTy, 0, "RP", BB);
    Instruction *RetLoad = new LoadInst(RetTy, RetAlloca, "", BB);
    ReturnInst::Create(Context, RetLoad, BB);
  } else {
    ReturnInst::Create(Context, BB);
  }

  return F;
}
Function *RandomIRBuilder::createFunctionDefinition(Module &M,
                                                    uint64_t MinArgNum,
                                                    uint64_t MaxArgNum) {
  return createFunctionDefinition(
      M, uniform<uint64_t>(Rand, MinArgNum, MaxArgNum));
}