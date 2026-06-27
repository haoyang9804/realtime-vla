from collections.abc import MutableMapping

import torch
import torch.nn.functional as F


HEAD_DIM = 256
NUM_Q_HEADS = 8
ATTN_SCALE = HEAD_DIM**-0.5
TensorMap = MutableMapping[str, torch.Tensor]


def ensure_sdpa_buffers(
    buffers: TensorMap,
    encoder_seq_len: int,
    decoder_seq_len: int,
) -> None:
    if "encoder_key_positions" not in buffers:
        buffers["encoder_key_positions"] = torch.arange(
            encoder_seq_len,
            dtype=torch.int32,
            device="cuda",
        )
    if "decoder_key_positions" not in buffers:
        buffers["decoder_key_positions"] = torch.arange(
            encoder_seq_len + decoder_seq_len,
            dtype=torch.int32,
            device="cuda",
        )


def sdpa_mqa_encoder(
    buffers: TensorMap,
    layer_idx: int,
    encoder_seq_len: int,
) -> None:
    _sdpa_mqa_attention(
        q_flat=buffers["encoder_Q"],
        k=buffers["encoder_K"][layer_idx, :encoder_seq_len],
        v=buffers["encoder_V"][layer_idx, :encoder_seq_len],
        key_positions=buffers["encoder_key_positions"],
        valid_prefix_len=buffers["valid_encoder_len"],
        out_flat=buffers["encoder_ctx_buf"],
        query_tokens=encoder_seq_len,
        prefix_keys=encoder_seq_len,
        suffix_keys=0,
    )


def sdpa_mqa_decoder(
    buffers: TensorMap,
    layer_idx: int,
    encoder_seq_len: int,
    decoder_seq_len: int,
) -> None:
    total_keys = encoder_seq_len + decoder_seq_len
    _sdpa_mqa_attention(
        q_flat=buffers["decoder_q_buf"],
        k=buffers["encoder_K"][layer_idx, :total_keys],
        v=buffers["encoder_V"][layer_idx, :total_keys],
        key_positions=buffers["decoder_key_positions"],
        valid_prefix_len=buffers["valid_encoder_len"],
        out_flat=buffers["decoder_q_buf"],
        query_tokens=decoder_seq_len,
        prefix_keys=encoder_seq_len,
        suffix_keys=decoder_seq_len,
    )


def _sdpa_mqa_attention(
    q_flat: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_positions: torch.Tensor,
    valid_prefix_len: torch.Tensor,
    out_flat: torch.Tensor,
    query_tokens: int,
    prefix_keys: int,
    suffix_keys: int,
) -> None:
    total_keys = prefix_keys + suffix_keys
    q = q_flat.view(query_tokens, NUM_Q_HEADS, HEAD_DIM).permute(1, 0, 2).unsqueeze(0)
    k_heads = k[:total_keys].view(1, 1, total_keys, HEAD_DIM)
    v_heads = v[:total_keys].view(1, 1, total_keys, HEAD_DIM)
    active_keys = (key_positions[:total_keys] < valid_prefix_len) | (
        key_positions[:total_keys] >= prefix_keys
    )
    attn = F.scaled_dot_product_attention(
        q,
        k_heads,
        v_heads,
        attn_mask=active_keys.view(1, 1, 1, total_keys),
        scale=ATTN_SCALE,
        enable_gqa=True,
    )
    out_flat.copy_(
        attn.squeeze(0)
        .permute(1, 0, 2)
        .contiguous()
        .view(query_tokens * NUM_Q_HEADS, HEAD_DIM)
    )
