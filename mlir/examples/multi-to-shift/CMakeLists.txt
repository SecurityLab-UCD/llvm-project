set(LLVM_LINK_COMPONENTS
  Support
  )


set(LIBS
  MLIRParser
  MLIRSupport
  MLIRIR
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
list(APPEND LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  MLIRPass
  MLIRArithDialect
  MLIRTransforms
  MLIRDialect
  MLIROptLib
  MLIRRewrite
)
add_mlir_tool(mlir-multi-to-shift-opt
  mlir-multi-to-shift-opt.cpp
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  ${LIBS}
  )
target_link_libraries(mlir-multi-to-shift-opt PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-multi-to-shift-opt)
mlir_check_all_link_libraries(mlir-multi-to-shift-opt)
