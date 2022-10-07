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

enum SourceType {
  PrevValueInBlock,
  FunctionArgument,
  ValueInDominator,
  GlobalStaticVari,
  NewSource,
  EndOfValueSrouce,
};

std::vector<BasicBlock *> getDominators(BasicBlock *BB) {
  std::vector<BasicBlock *> ret;
  DominatorTree DT(*BB->getParent());
  DomTreeNode *Node = DT[BB]->getIDom();
  while (Node) {
    ret.push_back(Node->getBlock());
    // Get parent block.
    Node = Node->getIDom();
  }
  return ret;
}

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
                                           SourcePred Pred) {
  auto MatchesPred = [&Srcs, &Pred](Value *V) { return Pred.matches(Srcs, V); };
  std::vector<uint64_t> SrcTys;
  for (uint64_t i = 0; i < EndOfValueSrouce; i++) {
    SrcTys.push_back(i);
  }
  // Get a random permutation of different types.
  for (uint64_t i = EndOfValueSrouce - 1; i > 0; i--) {
    std::swap(SrcTys[i], SrcTys[Rand() % i]);
  }
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
      auto dominators = getDominators(&BB);
      auto BBRS = makeSampler(Rand, dominators);
      if (BBRS.isEmpty()) {
        continue;
      }
      BasicBlock *Dom = BBRS.getSelection();
      // Somehow I can't use iterators to init these vectors, it will have type
      // mismatch.
      std::vector<Instruction *> Insts;
      for (Instruction &I : *Dom) {
        Insts.push_back(&I);
      }
      auto RS = makeSampler(Rand, make_filter_range(Insts, MatchesPred));
      // Also consider choosing no source, meaning we want a new one.
      if (!RS.isEmpty()) {
        return RS.getSelection();
      }
      break;
    }
    case GlobalStaticVari: {
      continue;
    }
    case NewSource: {
      return newSource(BB, Insts, Srcs, Pred);
    }
    case EndOfValueSrouce: {
      llvm_unreachable("Shouldn't be here");
    }
    }
  }
  llvm_unreachable("Shouldn't be here");
}

Value *RandomIRBuilder::newSource(BasicBlock &BB, ArrayRef<Instruction *> Insts,
                                  ArrayRef<Value *> Srcs, SourcePred Pred) {
  // Generate some constants to choose from.
  auto RS = makeSampler<Value *>(Rand);
  RS.sample(Pred.generate(Srcs, KnownTypes));

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

  assert(!RS.isEmpty() && "Failed to generate sources");
  return RS.getSelection();
}

static bool isCompatibleReplacement(const Instruction *I, const Use &Operand,
                                    const Value *Replacement) {
  if (Operand->getType() != Replacement->getType())
    return false;
  switch (I->getOpcode()) {
  case Instruction::GetElementPtr:
  case Instruction::ExtractElement:
  case Instruction::ExtractValue:
    // TODO: We could potentially validate these, but for now just leave
    // indices alone.
    if (Operand.getOperandNo() >= 1)
      return false;
    break;
  case Instruction::InsertValue:
  case Instruction::InsertElement:
  case Instruction::ShuffleVector:
    if (Operand.getOperandNo() >= 2)
      return false;
    break;
  default:
    break;
  }
  return true;
}

void RandomIRBuilder::connectToSink(BasicBlock &BB,
                                    ArrayRef<Instruction *> Insts, Value *V) {
  auto RS = makeSampler<Use *>(Rand);
  for (auto &I : Insts) {
    if (isa<IntrinsicInst>(I))
      // TODO: Replacing operands of intrinsics would be interesting,
      // but there's no easy way to verify that a given replacement is
      // valid given that intrinsics can impose arbitrary constraints.
      continue;
    for (Use &U : I->operands())
      if (isCompatibleReplacement(I, U, V))
        RS.sample(&U, 1);
  }
  // Also consider choosing no sink, meaning we want a new one.
  RS.sample(nullptr, /*Weight=*/1);

  if (Use *Sink = RS.getSelection()) {
    User *U = Sink->getUser();
    unsigned OpNo = Sink->getOperandNo();
    U->setOperand(OpNo, V);
    return;
  }
  newSink(BB, Insts, V);
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

Function *RandomIRBuilder::createFunctionDeclaration(Module &M,
                                                     uint64_t ArgNum) {
  uint64_t TyIdx = uniform<uint64_t>(Rand, 0, KnownTypes.size() - 1);
  Type *RetType = KnownTypes[TyIdx];

  SmallVector<Type *, 2> Args;
  for (uint64_t i = 0; i < ArgNum; i++) {
    TyIdx = uniform<uint64_t>(Rand, 0, KnownTypes.size() - 1);
    Args.push_back(KnownTypes[TyIdx]);
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