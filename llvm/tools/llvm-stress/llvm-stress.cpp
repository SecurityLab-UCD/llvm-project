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

#include "mutator.h"

#include "llvm/FuzzMutate/IRMutator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include <fstream>
#include <string>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace llvm {

static cl::OptionCategory StressCategory("Stress Options");

static cl::opt<unsigned> SeedCL("seed", cl::desc("Seed used for randomness"),
                                cl::init(0), cl::cat(StressCategory));

static cl::opt<std::string>
    InputFilename("i", cl::desc("Mutate an existing module from file"),
                  cl::value_desc("filename"), cl::cat(StressCategory));

static cl::opt<std::string> OutputFilename("o",
                                           cl::desc("Override output filename"),
                                           cl::value_desc("filename"),
                                           cl::cat(StressCategory));

namespace {

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

  LLVMContext Context;

  constexpr int MAX_SIZE = 1048576; // 1 MiB
  std::streamsize Size = 0;
  std::vector<char> Buffer(MAX_SIZE);

  if (!InputFilename.empty()) {
    std::ifstream InFile(InputFilename, std::ios::binary | std::ios::ate);
    Size = InFile.tellg();
    InFile.seekg(0, std::ios::beg);
    if (!InFile.read(Buffer.data(), Size)) {
      errs() << "Error reading input file.\n";
      return 1;
    }
  }

  createISelMutator();
  unsigned Seed = SeedCL;
  if (SeedCL == 0) {
    // Replace default value with a more reasonable seed.
    srand(mix(clock(), time(NULL), getpid()));
    Seed = rand();
  }
  errs() << Seed << '\n';
  size_t NewSize =
      LLVMFuzzerCustomMutator((uint8_t *)Buffer.data(), Size, MAX_SIZE, Seed);

  std::unique_ptr<llvm::Module> M =
      llvm::parseModule((uint8_t *)Buffer.data(), NewSize, Context);

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

  // Output textual IR.
  M->print(Out->os(), nullptr);

  Out->keep();

  return 0;
}
