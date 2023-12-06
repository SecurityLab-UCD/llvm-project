module  {
  func.func @main(%arg0: i32, %arg1: i32) -> i32 {
    %c16_i32 = arith.constant 16 : i32
    %0 = arith.shli %arg0, %c16_i32 : i32
    %1 = arith.shli %0, %c16_i32 : i32
    return %1 : i32
  }
}
