from collections.abc import MutableMapping

import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func


HEAD_DIM = 256
NUM_Q_HEADS = 8
BLOCK_M = 16
BLOCK_N = 64
DECODER_SPLIT_K = 4
ATTN_SCALE = HEAD_DIM**-0.5
TensorMap = MutableMapping[str, torch.Tensor]


def ensure_official_flash_encoder_buffers(
    buffers: TensorMap,
    encoder_seq_len: int,
) -> None:
    if "official_flash_encoder_cu_q" not in buffers:
        buffers["official_flash_encoder_cu_q"] = torch.tensor(
            [0, encoder_seq_len],
            dtype=torch.int32,
            device="cuda",
        )
    if "official_flash_encoder_cu_k" not in buffers:
        buffers["official_flash_encoder_cu_k"] = torch.empty(
            2,
            dtype=torch.int32,
            device="cuda",
        )
        buffers["official_flash_encoder_cu_k"][0].fill_(0)


def prepare_official_flash_encoder_buffers(buffers: TensorMap) -> None:
    buffers["official_flash_encoder_cu_k"][1:2].copy_(buffers["valid_encoder_len"])


def ensure_split_flash_decoder_buffers(
    buffers: TensorMap,
    decoder_seq_len: int,
    num_splits: int = DECODER_SPLIT_K,
) -> None:
    total_queries = decoder_seq_len * NUM_Q_HEADS
    acc_shape = (num_splits, total_queries, HEAD_DIM)
    stats_shape = (num_splits, total_queries)
    if buffers.get("flash_decoder_split_acc") is None or tuple(buffers["flash_decoder_split_acc"].shape) != acc_shape:
        buffers["flash_decoder_split_acc"] = torch.empty(acc_shape, dtype=torch.float32, device="cuda")
    if buffers.get("flash_decoder_split_m") is None or tuple(buffers["flash_decoder_split_m"].shape) != stats_shape:
        buffers["flash_decoder_split_m"] = torch.empty(stats_shape, dtype=torch.float32, device="cuda")
    if buffers.get("flash_decoder_split_l") is None or tuple(buffers["flash_decoder_split_l"].shape) != stats_shape:
        buffers["flash_decoder_split_l"] = torch.empty(stats_shape, dtype=torch.float32, device="cuda")


def official_flash_mqa_encoder(
    buffers: TensorMap,
    layer_idx: int,
    encoder_seq_len: int,
) -> None:
    attn = flash_attn_varlen_func(
        buffers["encoder_Q"].view(encoder_seq_len, NUM_Q_HEADS, HEAD_DIM),
        buffers["encoder_K"][layer_idx, :encoder_seq_len].view(encoder_seq_len, 1, HEAD_DIM),
        buffers["encoder_V"][layer_idx, :encoder_seq_len].view(encoder_seq_len, 1, HEAD_DIM),
        buffers["official_flash_encoder_cu_q"],
        buffers["official_flash_encoder_cu_k"],
        encoder_seq_len,
        encoder_seq_len,
        dropout_p=0.0,
        softmax_scale=ATTN_SCALE,
        causal=False,
    )
    buffers["encoder_ctx_buf"].copy_(attn.contiguous().view(encoder_seq_len * NUM_Q_HEADS, HEAD_DIM))


@triton.jit
def _flash_mqa_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    valid_prefix_len_ptr,
    out_ptr,
    query_tokens: tl.constexpr,
    prefix_keys: tl.constexpr,
    suffix_keys: tl.constexpr,
    scale: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    head_dim: tl.constexpr,
    num_q_heads: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_d = tl.arange(0, head_dim)
    q_offsets = (offs_m[:, None] * num_q_heads + pid_h) * head_dim + offs_d[None, :]
    q = tl.load(q_ptr + q_offsets, mask=offs_m[:, None] < query_tokens, other=0.0)

    m_i = tl.full((block_m,), -3.4028234663852886e38, tl.float32)
    l_i = tl.zeros((block_m,), tl.float32)
    acc = tl.zeros((block_m, head_dim), tl.float32)
    valid_prefix_len = tl.load(valid_prefix_len_ptr).to(tl.int32)
    total_keys: tl.constexpr = prefix_keys + suffix_keys

    for start_n in range(0, total_keys, block_n):
        offs_n = start_n + tl.arange(0, block_n)
        k = tl.load(
            k_ptr + offs_n[:, None] * head_dim + offs_d[None, :],
            mask=offs_n[:, None] < total_keys,
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k)) * scale
        prefix_ok = offs_n < valid_prefix_len
        suffix_ok = offs_n >= prefix_keys
        key_ok = (offs_n < total_keys) & (prefix_ok | suffix_ok)
        scores = tl.where(
            (offs_m[:, None] < query_tokens) & key_ok[None, :],
            scores,
            -3.4028234663852886e38,
        )

        row_m = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, row_m)
        p = tl.exp(scores - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        v = tl.load(
            v_ptr + offs_n[:, None] * head_dim + offs_d[None, :],
            mask=offs_n[:, None] < total_keys,
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        out_ptr + q_offsets,
        acc.to(tl.bfloat16),
        mask=offs_m[:, None] < query_tokens,
    )


def flash_mqa_encoder(
    buffers: TensorMap,
    layer_idx: int,
    encoder_seq_len: int,
) -> None:
    _launch_flash_mqa(
        q_flat=buffers["encoder_Q"],
        k=buffers["encoder_K"][layer_idx, :encoder_seq_len],
        v=buffers["encoder_V"][layer_idx, :encoder_seq_len],
        valid_prefix_len=buffers["valid_encoder_len"],
        out_flat=buffers["encoder_ctx_buf"],
        query_tokens=encoder_seq_len,
        prefix_keys=encoder_seq_len,
        suffix_keys=0,
    )


def flash_mqa_decoder(
    buffers: TensorMap,
    layer_idx: int,
    encoder_seq_len: int,
    decoder_seq_len: int,
) -> None:
    total_keys = encoder_seq_len + decoder_seq_len
    _launch_flash_mqa(
        q_flat=buffers["decoder_q_buf"],
        k=buffers["encoder_K"][layer_idx, :total_keys],
        v=buffers["encoder_V"][layer_idx, :total_keys],
        valid_prefix_len=buffers["valid_encoder_len"],
        out_flat=buffers["decoder_q_buf"],
        query_tokens=decoder_seq_len,
        prefix_keys=encoder_seq_len,
        suffix_keys=decoder_seq_len,
    )


@triton.jit
def _flash_mqa_decoder_split_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    valid_prefix_len_ptr,
    split_acc_ptr,
    split_m_ptr,
    split_l_ptr,
    query_tokens: tl.constexpr,
    prefix_keys: tl.constexpr,
    suffix_keys: tl.constexpr,
    scale: tl.constexpr,
    num_splits: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    head_dim: tl.constexpr,
    num_q_heads: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_d = tl.arange(0, head_dim)
    q_offsets = (offs_m[:, None] * num_q_heads + pid_h) * head_dim + offs_d[None, :]
    q = tl.load(q_ptr + q_offsets, mask=offs_m[:, None] < query_tokens, other=0.0)

    total_keys: tl.constexpr = prefix_keys + suffix_keys
    keys_per_split: tl.constexpr = triton.cdiv(total_keys, num_splits)
    split_start = pid_s * keys_per_split
    split_end = tl.minimum(split_start + keys_per_split, total_keys)
    valid_prefix_len = tl.load(valid_prefix_len_ptr).to(tl.int32)

    m_i = tl.full((block_m,), -3.4028234663852886e38, tl.float32)
    l_i = tl.zeros((block_m,), tl.float32)
    acc = tl.zeros((block_m, head_dim), tl.float32)

    start_n = split_start
    while start_n < split_end:
        offs_n = start_n + tl.arange(0, block_n)
        in_split = offs_n < split_end
        k = tl.load(
            k_ptr + offs_n[:, None] * head_dim + offs_d[None, :],
            mask=(offs_n[:, None] < total_keys) & in_split[:, None],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k)) * scale
        prefix_ok = offs_n < valid_prefix_len
        suffix_ok = offs_n >= prefix_keys
        key_ok = (offs_n < total_keys) & in_split & (prefix_ok | suffix_ok)
        scores = tl.where(
            (offs_m[:, None] < query_tokens) & key_ok[None, :],
            scores,
            -3.4028234663852886e38,
        )

        row_m = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, row_m)
        p = tl.exp(scores - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        v = tl.load(
            v_ptr + offs_n[:, None] * head_dim + offs_d[None, :],
            mask=(offs_n[:, None] < total_keys) & in_split[:, None],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new
        start_n += block_n

    query_offsets = offs_m * num_q_heads + pid_h
    split_base = pid_s * query_tokens * num_q_heads
    tl.store(
        split_acc_ptr + (split_base + query_offsets[:, None]) * head_dim + offs_d[None, :],
        acc,
        mask=offs_m[:, None] < query_tokens,
    )
    tl.store(
        split_m_ptr + split_base + query_offsets,
        m_i,
        mask=offs_m < query_tokens,
    )
    tl.store(
        split_l_ptr + split_base + query_offsets,
        l_i,
        mask=offs_m < query_tokens,
    )


@triton.jit
def _flash_mqa_decoder_combine_kernel(
    split_acc_ptr,
    split_m_ptr,
    split_l_ptr,
    out_ptr,
    query_tokens: tl.constexpr,
    num_splits: tl.constexpr,
    block_m: tl.constexpr,
    head_dim: tl.constexpr,
    num_q_heads: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_d = tl.arange(0, head_dim)
    query_offsets = offs_m * num_q_heads + pid_h

    m_i = tl.full((block_m,), -3.4028234663852886e38, tl.float32)
    l_i = tl.zeros((block_m,), tl.float32)
    acc = tl.zeros((block_m, head_dim), tl.float32)

    for split_id in range(0, num_splits):
        split_base = split_id * query_tokens * num_q_heads
        m_s = tl.load(
            split_m_ptr + split_base + query_offsets,
            mask=offs_m < query_tokens,
            other=-3.4028234663852886e38,
        )
        l_s = tl.load(
            split_l_ptr + split_base + query_offsets,
            mask=offs_m < query_tokens,
            other=0.0,
        )
        acc_s = tl.load(
            split_acc_ptr + (split_base + query_offsets[:, None]) * head_dim + offs_d[None, :],
            mask=offs_m[:, None] < query_tokens,
            other=0.0,
        )
        m_new = tl.maximum(m_i, m_s)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_s - m_new)
        acc = acc * alpha[:, None] + acc_s * beta[:, None]
        l_i = l_i * alpha + l_s * beta
        m_i = m_new

    acc = acc / l_i[:, None]
    q_offsets = (offs_m[:, None] * num_q_heads + pid_h) * head_dim + offs_d[None, :]
    tl.store(
        out_ptr + q_offsets,
        acc.to(tl.bfloat16),
        mask=offs_m[:, None] < query_tokens,
    )


def flash_mqa_decoder_split(
    buffers: TensorMap,
    layer_idx: int,
    encoder_seq_len: int,
    decoder_seq_len: int,
    num_splits: int = DECODER_SPLIT_K,
) -> None:
    total_keys = encoder_seq_len + decoder_seq_len
    grid = (triton.cdiv(decoder_seq_len, BLOCK_M), NUM_Q_HEADS, num_splits)
    _flash_mqa_decoder_split_kernel[grid](
        buffers["decoder_q_buf"],
        buffers["encoder_K"][layer_idx, :total_keys],
        buffers["encoder_V"][layer_idx, :total_keys],
        buffers["valid_encoder_len"],
        buffers["flash_decoder_split_acc"],
        buffers["flash_decoder_split_m"],
        buffers["flash_decoder_split_l"],
        decoder_seq_len,
        encoder_seq_len,
        decoder_seq_len,
        ATTN_SCALE,
        num_splits,
        BLOCK_M,
        BLOCK_N,
        HEAD_DIM,
        NUM_Q_HEADS,
    )
    combine_grid = (triton.cdiv(decoder_seq_len, BLOCK_M), NUM_Q_HEADS)
    _flash_mqa_decoder_combine_kernel[combine_grid](
        buffers["flash_decoder_split_acc"],
        buffers["flash_decoder_split_m"],
        buffers["flash_decoder_split_l"],
        buffers["decoder_q_buf"],
        decoder_seq_len,
        num_splits,
        BLOCK_M,
        HEAD_DIM,
        NUM_Q_HEADS,
    )


def _launch_flash_mqa(
    q_flat: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_prefix_len: torch.Tensor,
    out_flat: torch.Tensor,
    query_tokens: int,
    prefix_keys: int,
    suffix_keys: int,
) -> None:
    grid = (triton.cdiv(query_tokens, BLOCK_M), NUM_Q_HEADS)
    _flash_mqa_attention_kernel[grid](
        q_flat,
        k,
        v,
        valid_prefix_len,
        out_flat,
        query_tokens,
        prefix_keys,
        suffix_keys,
        ATTN_SCALE,
        BLOCK_M,
        BLOCK_N,
        HEAD_DIM,
        NUM_Q_HEADS,
    )
