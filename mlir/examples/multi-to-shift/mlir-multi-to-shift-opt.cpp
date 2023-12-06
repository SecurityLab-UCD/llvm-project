#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// docs: https://mlir.llvm.org/docs/PatternRewriter/
// arith dialect: https://mlir.llvm.org/docs/Dialects/ArithOps/
// examples: https://github.com/llvm/llvm-project/tree/main/mlir/examples

namespace mlir {

using arith::ConstantIntOp;
using arith::ConstantOp;
using arith::ShLIOp;

/// MultiToShiftPattern translates shl(shl(...(x, c1), c2), ..., cn) to shl(x,
/// c1 + ... + cn), where x is an integer, and c1 to cn are all integer
/// constants.
///
/// - If there is only one shift, nothing happens.
/// - The shift left operations in the middle remain unchanged, the dead code
/// elminator will erase them.
/// - If the total shift amount exceeds max bitwidth, nothing happens. Constant
/// folding should erase this/these operations into constant 0.
struct MultiToShiftPattern : public OpRewritePattern<ShLIOp> {
  MultiToShiftPattern(mlir::MLIRContext *context)
      : OpRewritePattern<ShLIOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(ShLIOp op,
                                PatternRewriter &rewriter) const override {
    // We only operate on integer types
    IntegerType Ty = llvm::dyn_cast<IntegerType>(op.getType());
    if (!Ty)
      return failure();
    unsigned bitwidth = Ty.getWidth();

    uint64_t ShiftAmt = 0;
    unsigned ConstCnt = 0;
    // ShiftVar tracks the real register that is been shifted.
    ShLIOp CurrShL = op;
    ConstantIntOp ConstInt =
        CurrShL.getOperand(1).getDefiningOp<ConstantIntOp>();
    // The real value that is shifted.
    Value ShiftVar;

    // Trace back to find all constants that are related to this shift.
    while (ConstInt && CurrShL && ShiftAmt < bitwidth) {
      ConstCnt++;
      // bitwidth is unsigned and ShiftAmt is uint64_t, so this add will not
      // overflow.
      ShiftAmt += ConstInt.value();

      // Move to the next instruction
      Value lhs = CurrShL.getOperand(0);
      ShiftVar = lhs;
      CurrShL = lhs.getDefiningOp<ShLIOp>();
      ConstInt = (CurrShL)
                     ? CurrShL.getOperand(1).getDefiningOp<ConstantIntOp>()
                     : nullptr;
    }

    // Only one constant found means that there is only one shift, do nothing.
    if (ConstCnt < 2)
      return failure();

    // The shift amount is too large the result will be zero.
    // In that case, we do nothing and count on constant folding to clean things
    // up.
    if (ShiftAmt > bitwidth)
      return failure();

    // Create a new constant representing the sum.
    ConstantOp ShiftAmtConstant = rewriter.create<ConstantOp>(
        op.getLoc(), rewriter.getIntegerAttr(Ty, ShiftAmt));

    // Replace the shift amount with the new constant.
    ShLIOp NewShift =
        rewriter.create<ShLIOp>(op.getLoc(), ShiftVar, ShiftAmtConstant);
    rewriter.replaceOp(op, ValueRange({NewShift}));

    // We don't have to explictly erase dead code caused by this replace.
    // Dead code elimination pass will do it for us.

    // Indicate successful rewrite.
    return success();
  }
};

class MultiToShiftPass
    : public PassWrapper<MultiToShiftPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const final { return "instcombine"; }

  StringRef getDescription() const final {
    return "A simple pass to combine any consecutive constant shifts left into "
           "a single shift left";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MultiToShiftPattern>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

} // namespace mlir

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::PassRegistration<mlir::MultiToShiftPass>();

  mlir::DialectRegistry registry;
  registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Custom optimizer driver\n", registry));
}
