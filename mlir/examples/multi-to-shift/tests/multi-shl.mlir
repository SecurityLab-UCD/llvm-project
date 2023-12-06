module  {
  func.func @main(%arg0: i32, %arg1: i32) -> i32 {
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = arith.shli %arg0, %arg1 : i32
    %1 = arith.shli %0, %c8_i32 : i32
    %2 = arith.shli %1, %c4_i32 : i32
    %3 = arith.shli %2, %c4_i32 : i32
    %4 = arith.shli %2, %c8_i32 : i32
    %5 = arith.shli %3, %4 : i32
    return %5 : i32
  }
}