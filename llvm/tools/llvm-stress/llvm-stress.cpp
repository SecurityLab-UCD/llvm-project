//===- llvm-stress.cpp - Generate random LL files to stress-test LLVM -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This program is a utility that generates random .ll files to stress-test
// different components in LLVM.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/FuzzMutate/IRMutator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include <fstream>
#include <string>
#include <time.h>
#include <unistd.h>
#include <vector>

#define DEBUG_TYPE "llvm-stress"

namespace llvm {

static cl::OptionCategory StressCategory("Stress Options");

static cl::opt<unsigned> SeedCL("seed", cl::desc("Seed used for randomness"),
                                cl::init(0), cl::cat(StressCategory));

static cl::opt<unsigned>
    RepeatCL("repeat", cl::desc("Number of times to mutate the module"),
             cl::init(100), cl::value_desc("times"), cl::cat(StressCategory));

static cl::opt<std::string>
    InputFilename("i", cl::desc("Mutate an existing module from file"),
                  cl::value_desc("filename"), cl::cat(StressCategory));

static cl::opt<std::string> OutputFilename("o",
                                           cl::desc("Override output filename"),
                                           cl::value_desc("filename"),
                                           cl::cat(StressCategory));

static cl::opt<unsigned> MaxSizeCL(
    "max-size", cl::desc("Max size of mutated module (defaults to 1 MiB)"),
    cl::init(1048576), cl::value_desc("bytes"), cl::cat(StressCategory));

namespace {
void addVectorTypeGetters(std::vector<TypeGetter> &Types) {
  int VectorLength[] = {1, 2, 4, 8, 16, 32};
  std::vector<TypeGetter> BasicTypeGetters(Types);
  for (auto typeGetter : BasicTypeGetters) {
    for (int length : VectorLength) {
      Types.push_back([typeGetter, length](LLVMContext &C) {
        return VectorType::get(typeGetter(C), length, false);
      });
    }
  }
}

auto createCustomMutator() {
  std::vector<TypeGetter> Types{
      Type::getInt1Ty,  Type::getInt8Ty,  Type::getInt16Ty, Type::getInt32Ty,
      Type::getInt64Ty, Type::getFloatTy, Type::getDoubleTy};
  std::vector<TypeGetter> ScalarTypes = Types;

  addVectorTypeGetters(Types);

  TypeGetter OpaquePtrGetter = [](LLVMContext &C) {
    return PointerType::get(Type::getInt32Ty(C), 0);
  };
  Types.push_back(OpaquePtrGetter);

  // Copy scalar types to change distribution.
  for (int i = 0; i < 5; i++)
    Types.insert(Types.end(), ScalarTypes.begin(), ScalarTypes.end());

  std::vector<std::unique_ptr<IRMutationStrategy>> Strategies;
  std::vector<fuzzerop::OpDescriptor> Ops = InjectorIRStrategy::getDefaultOps();

  Strategies.push_back(std::make_unique<InjectorIRStrategy>(
      InjectorIRStrategy::getDefaultOps()));
  Strategies.push_back(std::make_unique<InstDeleterIRStrategy>());
  Strategies.push_back(std::make_unique<InstModificationIRStrategy>());
  Strategies.push_back(std::make_unique<InsertFunctionStrategy>());
  Strategies.push_back(std::make_unique<InsertCFGStrategy>());
  Strategies.push_back(std::make_unique<InsertPHIStrategy>());
  Strategies.push_back(std::make_unique<SinkInstructionStrategy>());
  Strategies.push_back(std::make_unique<ShuffleBlockStrategy>());

  return std::make_unique<IRMutator>(std::move(Types), std::move(Strategies));
}

// https://stackoverflow.com/questions/322938/recommended-way-to-initialize-srand
unsigned long mix(unsigned long a, unsigned long b, unsigned long c) {
  a = a - b;
  a = a - c;
  a = a ^ (c >> 13);
  b = b - c;
  b = b - a;
  b = b ^ (a << 8);
  c = c - a;
  c = c - b;
  c = c ^ (b >> 13);
  a = a - b;
  a = a - c;
  a = a ^ (c >> 12);
  b = b - c;
  b = b - a;
  b = b ^ (a << 16);
  c = c - a;
  c = c - b;
  c = c ^ (b >> 5);
  a = a - b;
  a = a - c;
  a = a ^ (c >> 3);
  b = b - c;
  b = b - a;
  b = b ^ (a << 10);
  c = c - a;
  c = c - b;
  c = c ^ (b >> 15);
  return c;
}

} // end anonymous namespace
} // end namespace llvm

int main(int argc, char **argv) {
  using namespace llvm;

  InitLLVM X(argc, argv);
  cl::HideUnrelatedOptions({&StressCategory, &getColorCategory()});
  cl::ParseCommandLineOptions(argc, argv, "llvm codegen stress-tester\n");

  if (RepeatCL == 0) {
    errs() << "Repeat count must be greater than zero.\n";
    return 1;
  }

  LLVMContext Context;
  std::unique_ptr<Module> M;
  if (!InputFilename.empty()) {
    SMDiagnostic Diagnostic;
    M = parseIRFile(InputFilename, Diagnostic, Context);
    if (!M) {
      Diagnostic.print(argv[0], errs());
      return 1;
    }
  } else {
    M = std::make_unique<Module>("M", Context);
  }

  if (M->size() > MaxSizeCL) {
    errs() << "Given module is larger than " << MaxSizeCL << " bytes.\n";
    return 1;
  }

  unsigned Seed = SeedCL;
  if (SeedCL.getNumOccurrences() == 0) {
    // Replace default value with a more reasonable seed.
    srand(mix(clock(), time(NULL), getpid()));
    Seed = rand();
  }
  LLVM_DEBUG(dbgs() << Seed << '\n');

  srand(Seed);
  auto Mutator = createCustomMutator();
  for (unsigned i = 0; i < RepeatCL; i++) {
    Mutator->mutateModule(*M, rand(), MaxSizeCL);
  }

  // Figure out what stream we are supposed to write to...
  std::unique_ptr<ToolOutputFile> Out;
  // Default to standard output.
  if (OutputFilename.empty())
    OutputFilename = "-";

  std::error_code EC;
  Out.reset(new ToolOutputFile(OutputFilename, EC, sys::fs::OF_None));
  if (EC) {
    errs() << EC.message() << '\n';
    return 1;
  }

  // Check that the generated module is accepted by the verifier.
  if (verifyModule(*M.get(), &errs()))
    report_fatal_error("Broken module found, compilation aborted!");

  // If extension matches, output bitcode
  if (StringRef(OutputFilename).ends_with_insensitive(".bc")) {
    WriteBitcodeToFile(*M, Out->os());
  } else { // defaults to textual IR
    M->print(Out->os(), nullptr);
  }

  Out->keep();

  return 0;
}
