#!/usr/bin/env python3
#===- fuse-gate-up.py ----------------------------------------------------===//
# 后处理: 把 decode5(MLP) 的 gate/up 两个 4480x1536 matmul_transpose_b
# 融合成一个 8960x1536 matmul(权重在 flat buffer 中相邻: offset 0 与
# 6881280)。数值等价, 板端 bench 实测 +5%(更长的连续 DRAM 流)。
# 用法: python3 fuse-gate-up.py <build_dir>/examples/BuddyTensorParallel
#===-------------------------------------------------------------------===//
import pathlib
import sys

SUBGRAPH_FUSED = """module {
  func.func @subgraph0_decode5(%arg0: tensor<8960x1536xf32>, %arg1: tensor<1536x4480xf32>, %arg2: tensor<1x1x1536xf32>) -> tensor<1x1536xf32> {
    %0 = tosa.const_shape  {values = dense<[1, 1536]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.reshape %arg2, %0 : (tensor<1x1x1536xf32>, !tosa.shape<2>) -> tensor<1x1536xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1x8960xf32>
    %2 = linalg.matmul_transpose_b {cast = #linalg.type_fn<cast_signed>} ins(%1, %arg0 : tensor<1x1536xf32>, tensor<8960x1536xf32>) outs(%cst : tensor<1x8960xf32>) -> tensor<1x8960xf32>
    %gs = tensor.extract_slice %2[0, 0] [1, 4480] [1, 1] : tensor<1x8960xf32> to tensor<1x4480xf32>
    %us = tensor.extract_slice %2[0, 4480] [1, 4480] [1, 1] : tensor<1x8960xf32> to tensor<1x4480xf32>
    %3 = tosa.const_shape  {values = dense<[1, 1, 4480]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %4 = tosa.reshape %gs, %3 : (tensor<1x4480xf32>, !tosa.shape<3>) -> tensor<1x1x4480xf32>
    %5 = tosa.sigmoid %4 : (tensor<1x1x4480xf32>) -> tensor<1x1x4480xf32>
    %6 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %7 = tosa.mul %4, %5, %6 : (tensor<1x1x4480xf32>, tensor<1x1x4480xf32>, tensor<1xi8>) -> tensor<1x1x4480xf32>
    %11 = tosa.const_shape  {values = dense<[1, 1, 4480]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %12 = tosa.reshape %us, %11 : (tensor<1x4480xf32>, !tosa.shape<3>) -> tensor<1x1x4480xf32>
    %13 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %14 = tosa.mul %7, %12, %13 : (tensor<1x1x4480xf32>, tensor<1x1x4480xf32>, tensor<1xi8>) -> tensor<1x1x4480xf32>
    %15 = tosa.const_shape  {values = dense<[1, 4480]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %16 = tosa.reshape %14, %15 : (tensor<1x1x4480xf32>, !tosa.shape<2>) -> tensor<1x4480xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1x1536xf32>
    %17 = linalg.matmul_transpose_b {cast = #linalg.type_fn<cast_signed>} ins(%16, %arg1 : tensor<1x4480xf32>, tensor<1536x4480xf32>) outs(%cst_1 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    return %17 : tensor<1x1536xf32>
  }
}
"""

FORWARD_FUSED = """module {
  func.func private @subgraph0_decode5(memref<8960x1536xf32, strided<[1536, 1], offset: ?>>, memref<1536x4480xf32, strided<[4480, 1], offset: ?>>, memref<1x1x1536xf32, strided<[1536, 1536, 1], offset: ?>>) -> memref<1x1536xf32>
  func.func @forward_decode5(%arg0: memref<20643840xf32>, %arg1: memref<1x1x1536xf32>) -> memref<1x1536xf32> {
    %subview = memref.subview %arg0[0] [13762560] [1] : memref<20643840xf32> to memref<13762560xf32>
    %expand_shape = memref.expand_shape %subview [[0, 1]] output_shape [8960, 1536] : memref<13762560xf32> into memref<8960x1536xf32>
    %subview_2 = memref.subview %arg0[13762560] [6881280] [1] : memref<20643840xf32> to memref<6881280xf32, strided<[1], offset: 13762560>>
    %expand_shape_3 = memref.expand_shape %subview_2 [[0, 1]] output_shape [1536, 4480] : memref<6881280xf32, strided<[1], offset: 13762560>> into memref<1536x4480xf32, strided<[4480, 1], offset: 13762560>>
    %cast = memref.cast %expand_shape : memref<8960x1536xf32> to memref<8960x1536xf32, strided<[1536, 1], offset: ?>>
    %cast_5 = memref.cast %expand_shape_3 : memref<1536x4480xf32, strided<[4480, 1], offset: 13762560>> to memref<1536x4480xf32, strided<[4480, 1], offset: ?>>
    %cast_6 = memref.cast %arg1 : memref<1x1x1536xf32> to memref<1x1x1536xf32, strided<[1536, 1536, 1], offset: ?>>
    %0 = call @subgraph0_decode5(%cast, %cast_5, %cast_6) : (memref<8960x1536xf32, strided<[1536, 1], offset: ?>>, memref<1536x4480xf32, strided<[4480, 1], offset: ?>>, memref<1x1x1536xf32, strided<[1536, 1536, 1], offset: ?>>) -> memref<1x1536xf32>
    return %0 : memref<1x1536xf32>
  }
}
"""


def main() -> None:
    out_dir = pathlib.Path(sys.argv[1])
    sub = out_dir / "subgraph0_decode5.mlir"
    fwd = out_dir / "forward_decode5.mlir"

    sub_text = sub.read_text()
    # 守卫: 确认生产形状与权重偏移符合预期,否则不融合。
    if "tensor<4480x1536xf32>" not in sub_text or "tensor<1536x4480xf32>" not in sub_text:
        print("[fuse-gate-up] subgraph pattern mismatch, skip")
        return
    fwd_text = fwd.read_text()
    if "[6881280]" not in fwd_text or "[13762560]" not in fwd_text:
        print("[fuse-gate-up] forward pattern mismatch, skip")
        return

    sub.write_text(SUBGRAPH_FUSED)
    fwd.write_text(FORWARD_FUSED)
    print("[fuse-gate-up] decode5 gate+up fused OK")


if __name__ == "__main__":
    main()
