import triton
import triton.language as tl

from pi0_infer import (
    layer_norm_matmul_n256_1152_2048_bias,
    matmul_small_bias,
    rms_norm_kernel,
    vision_encoder,
)
from pi05_infer import (
    Pi05Inference,
    adarms_norm_style_proj,
)


MATMUL_CONFIGS = [
    triton.Config(
        {"BLOCK_SIZE_N": 16, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 16, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_K": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_K": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 128},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 128},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 128},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 128},
        num_warps=4,
        num_stages=3,
    ),
]

SMALL_OUTPUT_MATMUL_CONFIGS = [
    triton.Config(
        {"BLOCK_SIZE_N": 16, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 128},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 16, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 128},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 128},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 16, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 256},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 16, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 256},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 256},
        num_warps=4,
        num_stages=3,
    ),
]

GATE_CONFIGS = [
    triton.Config(
        {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_K": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 128},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
]

ROPE_QKV_CONFIGS = [
    triton.Config(
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128},
        num_warps=4,
        num_stages=3,
    ),
]

ATTN_QK_CONFIGS = [
    triton.Config(
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128},
        num_warps=4,
        num_stages=3,
    ),
]

SOFTMAX_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 1, "BLOCK_SIZE": 1024}, num_warps=4),
    triton.Config({"BLOCK_SIZE_M": 2, "BLOCK_SIZE": 1024}, num_warps=4),
    triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE": 1024}, num_warps=4),
]


@triton.autotune(configs=MATMUL_CONFIGS + SMALL_OUTPUT_MATMUL_CONFIGS, key=["seq_len", "features", "hidden"])
@triton.jit
def autotuned_matmul_small_bias_kernel(
    inp_ptr,
    weight_ptr,
    out_ptr,
    bias_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc += tl.load(
            bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=(j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden,
            other=0.0,
        )
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len)
                & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0,
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features)
                & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0,
            )
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len)
            & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
        )


@triton.autotune(
    configs=MATMUL_CONFIGS + SMALL_OUTPUT_MATMUL_CONFIGS,
    key=["seq_len", "features", "hidden"],
    restore_value=["out_ptr"],
)
@triton.jit
def autotuned_matmul_small_bias_res_kernel(
    inp_ptr,
    weight_ptr,
    out_ptr,
    bias_ptr,
    res_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.load(
            res_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len)
            & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0,
        ).to(tl.float32)
        acc += tl.load(
            bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=(j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden,
            other=0.0,
        )
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len)
                & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0,
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features)
                & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0,
            )
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len)
            & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
        )


@triton.autotune(configs=MATMUL_CONFIGS, key=["seq_len", "features", "hidden"])
@triton.jit
def autotuned_matmul_small_kernel(
    inp_ptr,
    weight_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len)
                & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0,
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features)
                & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0,
            )
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len)
            & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
        )


@triton.autotune(
    configs=MATMUL_CONFIGS,
    key=["seq_len", "features", "hidden"],
    restore_value=["out_ptr"],
)
@triton.jit
def autotuned_matmul_small_res_kernel(
    inp_ptr,
    weight_ptr,
    out_ptr,
    res_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.load(
            res_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len)
            & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0,
        ).to(tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len)
                & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0,
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features)
                & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0,
            )
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len)
            & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
        )


@triton.autotune(
    configs=MATMUL_CONFIGS,
    key=["seq_len", "features", "hidden"],
    restore_value=["out_ptr"],
)
@triton.jit
def autotuned_matmul_small_res_gate_kernel(
    inp_ptr,
    weight_ptr,
    out_ptr,
    res_ptr,
    gate_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.load(
            res_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len)
            & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0,
        ).to(tl.float32)
        matmul_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len)
                & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0,
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features)
                & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0,
            )
            matmul_acc = tl.dot(x, w, matmul_acc)
        gate = tl.load(
            gate_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len)
            & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0,
        ).to(tl.float32)
        acc += matmul_acc * gate
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len)
            & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
        )


@triton.autotune(configs=GATE_CONFIGS, key=["seq_len", "features", "hidden"])
@triton.jit
def autotuned_matmul_small_gate_kernel(
    inp_ptr,
    weight1_ptr,
    weight2_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid1 = tl.program_id(axis=0)
    psize1 = tl.num_programs(axis=0)
    pid2 = tl.program_id(axis=1)
    psize2 = tl.num_programs(axis=1)
    for i in range(pid1 * BLOCK_SIZE_N, seq_len, psize1 * BLOCK_SIZE_N):
        for j in range(pid2 * BLOCK_SIZE_M, hidden, psize2 * BLOCK_SIZE_M):
            acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
            acc2 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
            for k in range(0, features, BLOCK_SIZE_K):
                x = tl.load(
                    inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * features + k + tl.arange(0, BLOCK_SIZE_K),
                    mask=i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len,
                    other=0.0,
                )
                w = tl.load(
                    weight1_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M),
                )
                acc = tl.dot(x, w, acc)
                w2 = tl.load(
                    weight2_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M),
                )
                acc2 = tl.dot(x, w2, acc2)
            acc = acc * tl.sigmoid(1.5957691216057308 * acc * (1 + 0.044715 * acc * acc))
            acc = (acc * acc2).to(tl.bfloat16)
            tl.store(
                out_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M),
                acc,
                mask=i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len,
            )


@triton.autotune(configs=ROPE_QKV_CONFIGS, key=["seq_len", "features", "head_dim", "num_heads", "CAST_BEFORE_ROPE"])
@triton.jit
def autotuned_matmul_rope_qkv_kernel(
    inp_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    head_dim: tl.constexpr,
    num_heads: tl.constexpr,
    weight_qkv_ptr,
    rope_weights_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    CAST_BEFORE_ROPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    grid_m = triton.cdiv(seq_len, BLOCK_SIZE_M)
    grid_n = triton.cdiv((num_heads + 2) * head_dim, BLOCK_SIZE_N)
    assert head_dim % BLOCK_SIZE_N == 0
    while pid < grid_m * grid_n:
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        start_i = pid_m * BLOCK_SIZE_M
        start_j = pid_n * BLOCK_SIZE_N
        offs_i = start_i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = start_j + tl.arange(0, BLOCK_SIZE_N)[None, :]
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            offs_k = k + tl.arange(0, BLOCK_SIZE_K)
            x = tl.load(
                inp_ptr + offs_i * features + offs_k[None, :],
                mask=(offs_i < seq_len) & (offs_k[None, :] < features),
                other=0.0,
            )
            w = tl.load(
                weight_qkv_ptr + offs_k[:, None] * ((num_heads + 2) * head_dim) + offs_j,
                mask=(offs_k[:, None] < features) & (offs_j < (num_heads + 2) * head_dim),
                other=0.0,
            )
            acc = tl.dot(x, w, acc)
        if CAST_BEFORE_ROPE:
            acc = acc.to(tl.bfloat16)
        if start_j < (num_heads + 1) * head_dim:
            x0, x1 = tl.split(acc.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x_cossin = tl.load(
                rope_weights_ptr + offs_i * head_dim + offs_j % head_dim,
                mask=offs_i < seq_len,
                other=0.0,
            )
            x_cos, x_sin = tl.split(x_cossin.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x0_ = x0 * x_cos - x1 * x_sin
            x1_ = x1 * x_cos + x0 * x_sin
            acc = tl.interleave(x0_, x1_)
        acc = acc.to(tl.bfloat16)
        if start_j < num_heads * head_dim:
            out_ptr = q_ptr
            out_stride = num_heads * head_dim
        elif start_j < (num_heads + 1) * head_dim:
            out_ptr = k_ptr
            out_stride = head_dim
        else:
            out_ptr = v_ptr
            out_stride = head_dim
        tl.store(
            out_ptr + offs_i * out_stride + offs_j % out_stride,
            acc,
            mask=(offs_i < seq_len) & (offs_j < (num_heads + 2) * head_dim),
        )
        pid += psize


@triton.autotune(configs=ATTN_QK_CONFIGS, key=["M", "N", "K"])
@triton.jit
def autotuned_matmul_abt_scale_kernel(
    q_ptr,
    k_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    scale_factor: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    grid_m = triton.cdiv(M, BLOCK_SIZE_M)
    grid_n = triton.cdiv(N, BLOCK_SIZE_N)
    while pid < grid_m * grid_n:
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        offs_i = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_j = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_SIZE_K):
            offs_k = k + tl.arange(0, BLOCK_SIZE_K)
            x = tl.load(q_ptr + offs_i[:, None] * K + offs_k[None, :], mask=offs_k[None, :] < K, other=0.0)
            w = tl.load(k_ptr + offs_j[:, None] * K + offs_k[None, :], mask=offs_k[None, :] < K, other=0.0)
            acc = tl.dot(x, tl.trans(w), acc)
        acc = acc * scale_factor
        tl.store(
            out_ptr + offs_i[:, None] * N + offs_j[None, :],
            acc.to(tl.bfloat16),
            mask=(offs_i[:, None] < M) & (offs_j[None, :] < N),
        )
        pid += psize


@triton.autotune(configs=SOFTMAX_CONFIGS, key=["queries", "keys"])
@triton.jit
def autotuned_softmax_masklen_kernel(
    inp_ptr,
    queries: tl.constexpr,
    keys: tl.constexpr,
    valid_keys_len_ptr,
    out_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    big_neg = -2.3819763e38
    assert BLOCK_SIZE >= keys
    valid_keys_len = tl.load(valid_keys_len_ptr).to(tl.int32)
    valid_keys_len = tl.maximum(0, tl.minimum(valid_keys_len, keys))
    for i in range(pid * BLOCK_SIZE_M, queries, psize * BLOCK_SIZE_M):
        offs_i = i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = tl.arange(0, BLOCK_SIZE)[None, :]
        attn_mask = (offs_i < queries) & (offs_j < keys) & (offs_j < valid_keys_len)
        vals = tl.load(inp_ptr + offs_i * keys + offs_j, mask=attn_mask, other=big_neg)
        vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))
        vals = vals / tl.sum(vals, axis=1, keep_dims=True, dtype=tl.float32)
        tl.store(
            out_ptr + offs_i * keys + offs_j,
            vals.to(tl.bfloat16),
            mask=(offs_i < queries) & (offs_j < keys),
        )


@triton.autotune(configs=SOFTMAX_CONFIGS, key=["queries", "keys_prefix", "keys_suffix"])
@triton.jit
def autotuned_softmax_prefix_suffix_kernel(
    inp_ptr,
    queries: tl.constexpr,
    keys_prefix: tl.constexpr,
    keys_suffix: tl.constexpr,
    valid_prefix_len_ptr,
    out_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    big_neg = -2.3819763e38
    total_keys: tl.constexpr = keys_prefix + keys_suffix
    assert BLOCK_SIZE >= total_keys
    valid_prefix_len = tl.load(valid_prefix_len_ptr).to(tl.int32)
    valid_prefix_len = tl.maximum(0, tl.minimum(valid_prefix_len, keys_prefix))
    for i in range(pid * BLOCK_SIZE_M, queries, psize * BLOCK_SIZE_M):
        offs_i = i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = tl.arange(0, BLOCK_SIZE)[None, :]
        in_bounds = (offs_i < queries) & (offs_j < total_keys)
        is_prefix = offs_j < keys_prefix
        prefix_ok = is_prefix & (offs_j < valid_prefix_len)
        suffix_ok = ~is_prefix
        attn_mask = in_bounds & (prefix_ok | suffix_ok)
        vals = tl.load(inp_ptr + offs_i * total_keys + offs_j, mask=attn_mask, other=big_neg)
        vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))
        vals = vals / tl.sum(vals, axis=1, keep_dims=True, dtype=tl.float32)
        tl.store(out_ptr + offs_i * total_keys + offs_j, vals.to(tl.bfloat16), mask=in_bounds)


def _grid_1d(block_n, block_m, seq_len, hidden):
    return (triton.cdiv(seq_len, block_n) * triton.cdiv(hidden, block_m),)


def matmul_k_32_1024_bias_autotune(x, weight, bias, out):
    seq_len = x.shape[0]
    grid = lambda meta: _grid_1d(meta["BLOCK_SIZE_N"], meta["BLOCK_SIZE_M"], seq_len, 1024)
    autotuned_matmul_small_bias_kernel[grid](
        x,
        weight,
        out,
        bias,
        seq_len=seq_len,
        features=32,
        hidden=1024,
    )


def matmul_k8_n_256_autotune(x, v, out):
    total_queries = x.shape[0]
    total_keys = v.shape[0]
    grid = lambda meta: _grid_1d(meta["BLOCK_SIZE_N"], meta["BLOCK_SIZE_M"], total_queries, 256)
    autotuned_matmul_small_kernel[grid](
        x,
        v,
        out,
        seq_len=total_queries,
        features=total_keys,
        hidden=256,
    )


def matmul_n_2048_2048_res_autotune(x, weight, out):
    seq_len = x.shape[0]
    grid = lambda meta: _grid_1d(meta["BLOCK_SIZE_N"], meta["BLOCK_SIZE_M"], seq_len, 2048)
    autotuned_matmul_small_res_kernel[grid](
        x,
        weight,
        out,
        out,
        seq_len=seq_len,
        features=2048,
        hidden=2048,
    )


def matmul_n_16384_2048_res_autotune(x, weight, out):
    seq_len = x.shape[0]
    grid = lambda meta: _grid_1d(meta["BLOCK_SIZE_N"], meta["BLOCK_SIZE_M"], seq_len, 2048)
    autotuned_matmul_small_res_kernel[grid](
        x,
        weight,
        out,
        out,
        seq_len=seq_len,
        features=16384,
        hidden=2048,
    )


def matmul_k_2048_1024_gate_autotune(x, weight, out, gate):
    seq_len = x.shape[0]
    grid = lambda meta: _grid_1d(meta["BLOCK_SIZE_N"], meta["BLOCK_SIZE_M"], seq_len, 1024)
    autotuned_matmul_small_res_gate_kernel[grid](
        x,
        weight,
        out,
        out,
        gate,
        seq_len=seq_len,
        features=2048,
        hidden=1024,
    )


def matmul_k_4096_1024_gate_autotune(x, weight, out, gate):
    seq_len = x.shape[0]
    grid = lambda meta: _grid_1d(meta["BLOCK_SIZE_N"], meta["BLOCK_SIZE_M"], seq_len, 1024)
    autotuned_matmul_small_res_gate_kernel[grid](
        x,
        weight,
        out,
        out,
        gate,
        seq_len=seq_len,
        features=4096,
        hidden=1024,
    )


def matmul_small_gate_autotune(x, weight1, weight2, out, seq_len, features, hidden):
    grid = lambda meta: (triton.cdiv(seq_len, meta["BLOCK_SIZE_N"]), triton.cdiv(hidden, meta["BLOCK_SIZE_M"]))
    autotuned_matmul_small_gate_kernel[grid](
        x,
        weight1,
        weight2,
        out,
        seq_len,
        features,
        hidden,
    )


def rms_matmul_n_2048_16384_gate_autotune(x, weight1, weight2, out, x_norm):
    seq_len = x.shape[0]
    rms_norm_kernel[(seq_len,)](x, x_norm, seq_len, 2048)
    matmul_small_gate_autotune(x_norm, weight1, weight2, out, seq_len, 2048, 16384)


def rms_matmul_n_2048_2560_qkv_rope_autotune(x, weight_qkv, rope_weight, q, k, v, x_norm):
    seq_len = x.shape[0]
    rms_norm_kernel[(seq_len,)](x, x_norm, seq_len, 2048)
    grid = lambda meta: (
        triton.cdiv(seq_len, meta["BLOCK_SIZE_M"])
        * triton.cdiv(2560, meta["BLOCK_SIZE_N"]),
    )
    autotuned_matmul_rope_qkv_kernel[grid](
        x_norm,
        seq_len,
        2048,
        256,
        8,
        weight_qkv,
        rope_weight,
        q,
        k,
        v,
        CAST_BEFORE_ROPE=True,
    )


def matmul_k_1024_2560_qkv_rope_autotune(x_normed, weight_qkv, rope_weight, q, k, v):
    seq_len = x_normed.shape[0]
    grid = lambda meta: (
        triton.cdiv(seq_len, meta["BLOCK_SIZE_M"])
        * triton.cdiv(2560, meta["BLOCK_SIZE_N"]),
    )
    autotuned_matmul_rope_qkv_kernel[grid](
        x_normed,
        seq_len,
        1024,
        256,
        8,
        weight_qkv,
        rope_weight,
        q,
        k,
        v,
        CAST_BEFORE_ROPE=False,
    )


def adarms_matmul_k_1024_32_bias_res_autotune(
    x,
    time_emb,
    mod_w,
    mod_b,
    x_normed,
    gate,
    style,
    weight,
    bias,
    out,
    res,
):
    adarms_norm_style_proj(x, time_emb, mod_w, mod_b, x_normed, gate, style)
    seq_len = x.shape[0]
    grid = lambda meta: _grid_1d(meta["BLOCK_SIZE_N"], meta["BLOCK_SIZE_M"], seq_len, 32)
    autotuned_matmul_small_bias_res_kernel[grid](
        x_normed,
        weight,
        out,
        bias,
        res,
        seq_len=seq_len,
        features=1024,
        hidden=32,
    )


def autotuned_matmul_abt_scale(q, k, out, m, n, k_dim, scale):
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_SIZE_M"])
        * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
    )
    autotuned_matmul_abt_scale_kernel[grid](q, k, out, m, n, k_dim, scale)


def softmax_masklen_autotune(inp, queries, keys, valid_keys_len, out):
    grid = lambda meta: (triton.cdiv(queries, meta["BLOCK_SIZE_M"]),)
    autotuned_softmax_masklen_kernel[grid](inp, queries, keys, valid_keys_len, out)


def softmax_prefix_suffix_autotune(inp, queries, prefix_keys, suffix_keys, valid_prefix_len, out):
    grid = lambda meta: (triton.cdiv(queries, meta["BLOCK_SIZE_M"]),)
    autotuned_softmax_prefix_suffix_kernel[grid](
        inp,
        queries,
        prefix_keys,
        suffix_keys,
        valid_prefix_len,
        out,
    )


def transformer_encoder_autotune(weights, buffers, encoder_seq_len):
    layer_norm_matmul_n256_1152_2048_bias(
        buffers["vision_x"],
        weights["vision_final_norm_w"],
        weights["vision_final_norm_b"],
        weights["encoder_multi_modal_projector_w"],
        weights["encoder_multi_modal_projector_b"],
        buffers["encoder_x"],
        buffers["vision_x_norm"],
    )
    for i in range(18):
        rms_matmul_n_2048_2560_qkv_rope_autotune(
            buffers["encoder_x"],
            weights["encoder_attn_qkv_w"][i],
            buffers["encoder_rope_weights"],
            buffers["encoder_Q"],
            buffers["encoder_K"][i, :encoder_seq_len],
            buffers["encoder_V"][i, :encoder_seq_len],
            buffers["encoder_x_norm"],
        )
        if i != 17:
            total_queries = buffers["encoder_Q"].shape[0]
            autotuned_matmul_abt_scale(
                buffers["encoder_Q"],
                buffers["encoder_K"][i, :encoder_seq_len],
                buffers["encoder_logits_buf"],
                total_queries,
                encoder_seq_len,
                256,
                256**-0.5,
            )
            softmax_masklen_autotune(
                buffers["encoder_logits_buf"],
                total_queries,
                encoder_seq_len,
                buffers["valid_encoder_len"],
                buffers["encoder_attn_buf"],
            )
            matmul_k8_n_256_autotune(
                buffers["encoder_attn_buf"],
                buffers["encoder_V"][i, :encoder_seq_len],
                buffers["encoder_ctx_buf"],
            )
            matmul_n_2048_2048_res_autotune(
                buffers["encoder_ctx_buf"].view(-1, 2048),
                weights["encoder_attn_o_w"][i],
                buffers["encoder_x"],
            )
            rms_matmul_n_2048_16384_gate_autotune(
                buffers["encoder_x"],
                weights["encoder_ffn_gate_w"][i],
                weights["encoder_ffn_up_w"][i],
                buffers["encoder_hidden"],
                buffers["encoder_x_norm"],
            )
            matmul_n_16384_2048_res_autotune(
                buffers["encoder_hidden"],
                weights["encoder_ffn_down_w"][i],
                buffers["encoder_x"],
            )


def transformer_decoder_autotune(weights, buffers, encoder_seq_len, num_steps=10):
    for step in range(num_steps):
        matmul_k_32_1024_bias_autotune(
            buffers["diffusion_noise"],
            weights["decoder_action_in_proj_w"],
            weights["decoder_action_in_proj_b"],
            buffers["decoder_x"],
        )
        seq_len = buffers["decoder_x"].shape[0]
        for i in range(18):
            adarms_norm_style_proj(
                buffers["decoder_x"],
                buffers["decoder_time_emb"][step],
                weights["decoder_pre_attn_norm_mod_w"][i],
                weights["decoder_pre_attn_norm_mod_b"][i],
                buffers["x_normed_buf"],
                buffers["gate_buf"],
                buffers["decoder_style_attn"][step, i],
            )
            matmul_k_1024_2560_qkv_rope_autotune(
                buffers["x_normed_buf"],
                weights["decoder_attn_qkv_w"][i],
                buffers["decoder_rope_weights"],
                buffers["decoder_q_buf"],
                buffers["encoder_K"][i, encoder_seq_len : encoder_seq_len + seq_len],
                buffers["encoder_V"][i, encoder_seq_len : encoder_seq_len + seq_len],
            )
            total_queries = buffers["decoder_q_buf"].shape[0]
            prefix_keys = encoder_seq_len
            suffix_keys = seq_len
            total_keys = prefix_keys + suffix_keys
            autotuned_matmul_abt_scale(
                buffers["decoder_q_buf"],
                buffers["encoder_K"][i, : encoder_seq_len + seq_len],
                buffers["decoder_logits_buf"],
                total_queries,
                total_keys,
                256,
                256**-0.5,
            )
            softmax_prefix_suffix_autotune(
                buffers["decoder_logits_buf"],
                total_queries,
                prefix_keys,
                suffix_keys,
                buffers["valid_encoder_len"],
                buffers["decoder_attn_buf"],
            )
            matmul_k8_n_256_autotune(
                buffers["decoder_attn_buf"],
                buffers["encoder_V"][i, : encoder_seq_len + seq_len],
                buffers["decoder_q_buf"],
            )
            matmul_k_2048_1024_gate_autotune(
                buffers["decoder_q_buf"].view(-1, 2048),
                weights["decoder_attn_o_w"][i],
                buffers["decoder_x"],
                buffers["gate_buf"],
            )
            adarms_norm_style_proj(
                buffers["decoder_x"],
                buffers["decoder_time_emb"][step],
                weights["decoder_pre_ffn_norm_mod_w"][i],
                weights["decoder_pre_ffn_norm_mod_b"][i],
                buffers["x_normed_buf"],
                buffers["gate_buf"],
                buffers["decoder_style_ffn"][step, i],
            )
            seq_len = buffers["decoder_x"].shape[0]
            matmul_small_gate_autotune(
                buffers["x_normed_buf"],
                weights["decoder_ffn_gate_w"][i],
                weights["decoder_ffn_up_w"][i],
                buffers["decoder_hidden"],
                seq_len,
                1024,
                4096,
            )
            matmul_k_4096_1024_gate_autotune(
                buffers["decoder_hidden"],
                weights["decoder_ffn_down_w"][i],
                buffers["decoder_x"],
                buffers["gate_buf"],
            )
        adarms_matmul_k_1024_32_bias_res_autotune(
            buffers["decoder_x"],
            buffers["decoder_time_emb"][step],
            weights["decoder_final_norm_mod_w"],
            weights["decoder_final_norm_mod_b"],
            buffers["x_normed_buf"],
            buffers["gate_buf"],
            buffers["decoder_style_final"][step],
            weights["decoder_action_out_proj_w"],
            weights["decoder_action_out_proj_b"],
            buffers["diffusion_noise"],
            buffers["diffusion_noise"],
        )


def pi05_model_autotune(weights, buffers, num_views, encoder_seq_len, num_steps=10):
    vision_encoder(weights, buffers, num_views)
    transformer_encoder_autotune(weights, buffers, encoder_seq_len)
    transformer_decoder_autotune(weights, buffers, encoder_seq_len, num_steps)


class Pi05AutotuneInference(Pi05Inference):
    def record_run(self):
        pi05_model_autotune(self.weights, self.buffers, self.num_views, self.encoder_seq_len)
