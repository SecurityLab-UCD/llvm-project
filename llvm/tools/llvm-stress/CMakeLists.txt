set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  Analysis
  AsmPrinter
  BitReader
  BitWriter
  Core
  FuzzMutate
  IRReader
  MC
  ScalarOpts
  Support
  Target
  )

add_llvm_tool(llvm-stress
  llvm-stress.cpp

  DEPENDS
  intrinsics_gen
  )
