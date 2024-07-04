//===-- HelloWorld.cpp - Example Transformations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/RemoveAttribute.h"

using namespace llvm;

PreservedAnalyses RemoveAttributePass::run(Module &M,
                                           ModuleAnalysisManager &MAM) {
  errs() << M.getName() << '\n';
  for (Function &F : M) {
    errs() << F.getName() << '\n';
    // Remove function attribute
    F.setAttributes({});
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        // Remove instruction metadata
        SmallVector<std::pair<unsigned, MDNode *>> AllMetadata;
        I.getAllMetadata(AllMetadata);
        for (auto p : AllMetadata) {
          I.setMetadata(p.first, nullptr);
        }
        continue;
      }
    }
  }
  return PreservedAnalyses::all();
}
