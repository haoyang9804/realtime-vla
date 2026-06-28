import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer


HEAD_DIM = 256
NUM_Q_HEADS = 8
ENCODER_WIDTH = 2048
DECODER_WIDTH = 1024
NUM_LAYERS = 18
NUM_VISION_LAYERS = 27
NUM_STEPS = 10

_COMPILE_ENABLED = os.environ.get("PI05_TORCH_COMPILE", "1") != "0"
_COMPILED = {}
_COMPILE_FAILED = set()


def _compiled_call(name: str, fn, *args):
    if not _COMPILE_ENABLED or name in _COMPILE_FAILED:
        return fn(*args)
    compiled = _COMPILED.get(name)
    if compiled is None:
        compiled = torch.compile(fn, mode="reduce-overhead", fullgraph=False)
        _COMPILED[name] = compiled
    try:
        return compiled(*args)
    except Exception as exc:
        _COMPILE_FAILED.add(name)
        print(f"torch.compile disabled for {name}: {type(exc).__name__}: {exc}")
        return fn(*args)


def _to_cuda_bf16(tensor: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    return tensor.to(device="cuda", dtype=torch.bfloat16, non_blocking=True)


def _linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    out = x @ weight
    if bias is not None:
        out = out + bias
    return out.to(torch.bfloat16)


def _gelu_approx(x: torch.Tensor) -> torch.Tensor:
    x_f = x.float()
    return (
        x_f
        * torch.sigmoid(1.5957691216057308 * x_f * (1.0 + 0.044715 * x_f * x_f))
    ).to(torch.bfloat16)


def _silu(x: torch.Tensor) -> torch.Tensor:
    return (x.float() * torch.sigmoid(x.float())).to(torch.bfloat16)


def _layer_norm(x: torch.Tensor, norm_w: torch.Tensor, norm_b: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(
        x.float(),
        (x.shape[-1],),
        norm_w.float(),
        norm_b.float(),
        eps=1e-5,
    ).to(torch.bfloat16)


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    factor = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    return (x.float() * factor).to(torch.bfloat16)


def _apply_rope_adjacent(x: torch.Tensor, rope_weights: torch.Tensor) -> torch.Tensor:
    seq_len = x.shape[0]
    x_pair = x.float().view(seq_len, -1, HEAD_DIM // 2, 2)
    rope_pair = rope_weights.float().view(seq_len, 1, HEAD_DIM // 2, 2)
    x0 = x_pair[..., 0]
    x1 = x_pair[..., 1]
    cos = rope_pair[..., 0]
    sin = rope_pair[..., 1]
    out = torch.stack((x0 * cos - x1 * sin, x1 * cos + x0 * sin), dim=-1)
    return out.reshape_as(x).to(torch.bfloat16)


def matmul_small_bias_torch(inp, weight, out, bias):
    out.copy_(_linear(inp, weight, bias))


def matmul_small_bias_res_torch(inp, weight, out, bias, res):
    out.copy_((res.float() + (inp @ weight).float() + bias.float()).to(torch.bfloat16))


def matmul_small_bias_silu_torch(inp, weight, out, bias):
    out[: inp.shape[0]].copy_(_silu(_linear(inp, weight, bias)))


def matmul_small_res_torch(inp, weight, out, res):
    out.copy_((res.float() + (inp @ weight).float()).to(torch.bfloat16))


def matmul_small_res_gate_torch(inp, weight, out, res, gate):
    out.copy_((res.float() + (inp @ weight).float() * gate.float()).to(torch.bfloat16))


def _matmul_small_gate_value(inp, weight1, weight2):
    gate = _linear(inp, weight1)
    up = _linear(inp, weight2)
    return (_gelu_approx(gate).float() * up.float()).to(torch.bfloat16)


def matmul_small_gate_torch(inp, weight1, weight2, out):
    out.copy_(_compiled_call("matmul_small_gate", _matmul_small_gate_value, inp, weight1, weight2))


def matmul_abt_scale_torch(q, k, out, scale):
    out.copy_(((q @ k.T) * scale).to(torch.bfloat16).to(torch.float32))


def softmax_masklen_torch(inp, valid_keys_len, out):
    keys = inp.shape[1]
    key_positions = torch.arange(keys, device=inp.device)
    mask = key_positions[None, :] < valid_keys_len.to(torch.int64)
    vals = inp.masked_fill(~mask, -2.3819763e38)
    out.copy_(torch.softmax(vals, dim=-1).to(torch.bfloat16))


def softmax_prefix_suffix_torch(inp, keys_prefix, keys_suffix, valid_prefix_len, out):
    total_keys = keys_prefix + keys_suffix
    key_positions = torch.arange(total_keys, device=inp.device)
    is_prefix = key_positions < keys_prefix
    active = (is_prefix & (key_positions < valid_prefix_len.to(torch.int64))) | (~is_prefix)
    vals = inp[:, :total_keys].masked_fill(~active[None, :], -2.3819763e38)
    out[:, :total_keys].copy_(torch.softmax(vals, dim=-1).to(torch.bfloat16))


def matmul_k8_n_256_torch(inp, v, out):
    out.copy_((inp @ v).to(torch.bfloat16))


def conv2d_embed_n256_1152_res_torch(images, patch_w, patch_b, pos_emb, out):
    nviews = images.shape[0]
    img_input = (
        images.view(nviews, 16, 14, 16, 14, 3)
        .permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(nviews, 256, 14 * 14 * 3)
    )
    flat_w = patch_w.view(14 * 14 * 3, 1152)
    out.copy_((_linear(img_input, flat_w, patch_b) + pos_emb.unsqueeze(0)).to(torch.bfloat16))


def layer_norm_qkv_matmul_n256_1152_3456_bias_torch(
    x, norm_w, norm_b, qkv_w, qkv_b, out, x_norm
):
    x_norm.copy_(_layer_norm(x, norm_w, norm_b))
    out.copy_(_linear(x_norm, qkv_w, qkv_b))


def matmul_n256_1152_1152_bias_res_torch(x, weight, bias, res, out):
    out.copy_((res.float() + (x @ weight).float() + bias.float()).to(torch.bfloat16))


def layer_norm_matmul_n256_1152_4304_bias_gelu_torch(
    x, norm_w, norm_b, weight, bias, out, x_norm
):
    x_norm.copy_(_layer_norm(x, norm_w, norm_b))
    out.copy_(_gelu_approx(_linear(x_norm, weight, bias)))


def matmul_n256_4304_1152_bias_res_torch(x, weight, bias, res, out):
    out.copy_((res.float() + (x @ weight).float() + bias.float()).to(torch.bfloat16))


def _attn_multi_key_value(qkv):
    qkv = qkv.view(-1, 256, 3, 16, 72).permute(0, 2, 3, 1, 4)
    q = qkv[:, 0]
    k = qkv[:, 1]
    v = qkv[:, 2]
    attn = F.scaled_dot_product_attention(q, k, v)
    return attn.transpose(1, 2).reshape(q.shape[0], 256, 1152).to(torch.bfloat16)


def attn_multi_key_torch(qkv):
    return _compiled_call("attn_multi_key", _attn_multi_key_value, qkv)


def vision_encoder_torch(weights, buffers, num_views):
    conv2d_embed_n256_1152_res_torch(
        buffers["observation_images_normalized"],
        weights["vision_patch_embedding_w"],
        weights["vision_patch_embedding_b"],
        weights["vision_position_embedding"],
        buffers["vision_x"],
    )

    for i in range(NUM_VISION_LAYERS):
        layer_norm_qkv_matmul_n256_1152_3456_bias_torch(
            buffers["vision_x"],
            weights["vision_pre_attn_norm_w"][i],
            weights["vision_pre_attn_norm_b"][i],
            weights["vision_attn_qkv_w"][i],
            weights["vision_attn_qkv_b"][i],
            buffers["vision_QKV"],
            buffers["vision_x_norm"],
        )
        attn = attn_multi_key_torch(buffers["vision_QKV"])
        matmul_n256_1152_1152_bias_res_torch(
            attn,
            weights["vision_attn_o_w"][i],
            weights["vision_attn_o_b"][i],
            buffers["vision_x"],
            buffers["vision_x"],
        )
        layer_norm_matmul_n256_1152_4304_bias_gelu_torch(
            buffers["vision_x"],
            weights["vision_pre_ffn_norm_w"][i],
            weights["vision_pre_ffn_norm_b"][i],
            weights["vision_ffn_up_w"][i],
            weights["vision_ffn_up_b"][i],
            buffers["vision_hidden"],
            buffers["vision_x_norm"],
        )
        matmul_n256_4304_1152_bias_res_torch(
            buffers["vision_hidden"],
            weights["vision_ffn_down_w"][i],
            weights["vision_ffn_down_b"][i],
            buffers["vision_x"],
            buffers["vision_x"],
        )


def layer_norm_matmul_n256_1152_2048_bias_torch(
    x, norm_w, norm_b, proj_w, proj_b, out, x_norm
):
    seq_len = x.shape[0] * 256
    x_norm.copy_(_layer_norm(x, norm_w, norm_b))
    out[:seq_len].copy_(_linear(x_norm.view(seq_len, 1152), proj_w, proj_b))


def _encoder_qkv_rope_value(x, weight_qkv, rope_weight):
    seq_len = x.shape[0]
    x_norm = _rms_norm(x)
    qkv = _linear(x_norm, weight_qkv)
    q, k, v = qkv.split([ENCODER_WIDTH, HEAD_DIM, HEAD_DIM], dim=-1)
    q = q.view(seq_len, NUM_Q_HEADS, HEAD_DIM)
    k = k.view(seq_len, 1, HEAD_DIM)
    q = _apply_rope_adjacent(q, rope_weight).view(seq_len * NUM_Q_HEADS, HEAD_DIM)
    k = _apply_rope_adjacent(k, rope_weight).view(seq_len, HEAD_DIM)
    return q, k, v, x_norm


def rms_matmul_n_2048_2560_qkv_rope_torch(x, weight_qkv, rope_weight, q_out, k_out, v_out, x_norm):
    q, k, v, x_norm_value = _compiled_call(
        "encoder_qkv_rope",
        _encoder_qkv_rope_value,
        x,
        weight_qkv,
        rope_weight,
    )
    q_out.copy_(q)
    k_out.copy_(k)
    v_out.copy_(v)
    x_norm.copy_(x_norm_value)


def matmul_n_2048_2048_res_torch(x, weight, out):
    matmul_small_res_torch(x, weight, out, out)


def rms_matmul_n_2048_16384_gate_torch(x, weight1, weight2, out, x_norm):
    x_norm.copy_(_rms_norm(x))
    matmul_small_gate_torch(x_norm, weight1, weight2, out)


def matmul_n_16384_2048_res_torch(x, weight, out):
    matmul_small_res_torch(x, weight, out, out)


def transformer_encoder_torch(weights, buffers, encoder_seq_len):
    layer_norm_matmul_n256_1152_2048_bias_torch(
        buffers["vision_x"],
        weights["vision_final_norm_w"],
        weights["vision_final_norm_b"],
        weights["encoder_multi_modal_projector_w"],
        weights["encoder_multi_modal_projector_b"],
        buffers["encoder_x"],
        buffers["vision_x_norm"],
    )
    for i in range(NUM_LAYERS):
        rms_matmul_n_2048_2560_qkv_rope_torch(
            buffers["encoder_x"],
            weights["encoder_attn_qkv_w"][i],
            buffers["encoder_rope_weights"],
            buffers["encoder_Q"],
            buffers["encoder_K"][i, :encoder_seq_len],
            buffers["encoder_V"][i, :encoder_seq_len],
            buffers["encoder_x_norm"],
        )
        if i != NUM_LAYERS - 1:
            total_queries = buffers["encoder_Q"].shape[0]
            total_keys = encoder_seq_len
            matmul_abt_scale_torch(
                buffers["encoder_Q"],
                buffers["encoder_K"][i, :encoder_seq_len],
                buffers["encoder_logits_buf"],
                HEAD_DIM**-0.5,
            )
            softmax_masklen_torch(
                buffers["encoder_logits_buf"][:total_queries, :total_keys],
                buffers["valid_encoder_len"],
                buffers["encoder_attn_buf"][:total_queries, :total_keys],
            )
            matmul_k8_n_256_torch(
                buffers["encoder_attn_buf"],
                buffers["encoder_V"][i, :encoder_seq_len],
                buffers["encoder_ctx_buf"],
            )
            matmul_n_2048_2048_res_torch(
                buffers["encoder_ctx_buf"].view(-1, ENCODER_WIDTH),
                weights["encoder_attn_o_w"][i],
                buffers["encoder_x"],
            )
            rms_matmul_n_2048_16384_gate_torch(
                buffers["encoder_x"],
                weights["encoder_ffn_gate_w"][i],
                weights["encoder_ffn_up_w"][i],
                buffers["encoder_hidden"],
                buffers["encoder_x_norm"],
            )
            matmul_n_16384_2048_res_torch(
                buffers["encoder_hidden"],
                weights["encoder_ffn_down_w"][i],
                buffers["encoder_x"],
            )


def matmul_k_32_1024_bias_torch(x, weight, bias, out):
    matmul_small_bias_torch(x, weight, out, bias)


def _adarms_norm_value(x, style):
    scale, shift, gate_style = style.split(DECODER_WIDTH, dim=-1)
    x_rms = _rms_norm(x)
    x_normed = (x_rms.float() * (1.0 + scale.float()) + shift.float()).to(torch.bfloat16)
    return x_normed, gate_style


def adarms_norm_style_proj_torch(x, time_emb, mod_w, mod_b, x_normed, gate, style):
    del time_emb, mod_w, mod_b
    x_normed_value, gate_style = _compiled_call("adarms_norm", _adarms_norm_value, x, style)
    x_normed.copy_(x_normed_value)
    gate.copy_(gate_style)


def _decoder_qkv_rope_value(x_normed, weight_qkv, rope_weight):
    seq_len = x_normed.shape[0]
    qkv = x_normed.float() @ weight_qkv.float()
    q, k, v = qkv.split([ENCODER_WIDTH, HEAD_DIM, HEAD_DIM], dim=-1)
    q = q.view(seq_len, NUM_Q_HEADS, HEAD_DIM)
    k = k.view(seq_len, 1, HEAD_DIM)
    q = _apply_rope_adjacent(q, rope_weight).view(seq_len * NUM_Q_HEADS, HEAD_DIM)
    k = _apply_rope_adjacent(k, rope_weight).view(seq_len, HEAD_DIM)
    return q, k, v.to(torch.bfloat16)


def matmul_k_1024_2560_qkv_rope_torch(x_normed, weight_qkv, rope_weight, q_out, k_out, v_out):
    q, k, v = _compiled_call("decoder_qkv_rope", _decoder_qkv_rope_value, x_normed, weight_qkv, rope_weight)
    q_out.copy_(q)
    k_out.copy_(k)
    v_out.copy_(v.to(torch.bfloat16))


def matmul_k_2048_1024_gate_torch(x, weight, out, gate):
    matmul_small_res_gate_torch(x, weight, out, out, gate)


def matmul_k_4096_1024_gate_torch(x, weight, out, gate):
    matmul_small_res_gate_torch(x, weight, out, out, gate)


def adarms_matmul_k_1024_32_bias_res_torch(
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
    adarms_norm_style_proj_torch(x, time_emb, mod_w, mod_b, x_normed, gate, style)
    matmul_small_bias_res_torch(x_normed, weight, out, bias, res)


def transformer_decoder_torch(weights, buffers, encoder_seq_len, num_steps=NUM_STEPS):
    for step in range(num_steps):
        matmul_k_32_1024_bias_torch(
            buffers["diffusion_noise"],
            weights["decoder_action_in_proj_w"],
            weights["decoder_action_in_proj_b"],
            buffers["decoder_x"],
        )
        seq_len = buffers["decoder_x"].shape[0]
        for i in range(NUM_LAYERS):
            adarms_norm_style_proj_torch(
                buffers["decoder_x"],
                buffers["decoder_time_emb"][step],
                weights["decoder_pre_attn_norm_mod_w"][i],
                weights["decoder_pre_attn_norm_mod_b"][i],
                buffers["x_normed_buf"],
                buffers["gate_buf"],
                buffers["decoder_style_attn"][step, i],
            )
            matmul_k_1024_2560_qkv_rope_torch(
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
            matmul_abt_scale_torch(
                buffers["decoder_q_buf"],
                buffers["encoder_K"][i, :total_keys],
                buffers["decoder_logits_buf"],
                HEAD_DIM**-0.5,
            )
            softmax_prefix_suffix_torch(
                buffers["decoder_logits_buf"][:total_queries, :total_keys],
                prefix_keys,
                suffix_keys,
                buffers["valid_encoder_len"],
                buffers["decoder_attn_buf"][:total_queries, :total_keys],
            )
            matmul_k8_n_256_torch(
                buffers["decoder_attn_buf"][:total_queries, :total_keys],
                buffers["encoder_V"][i, :total_keys],
                buffers["decoder_q_buf"],
            )
            matmul_k_2048_1024_gate_torch(
                buffers["decoder_q_buf"].view(-1, ENCODER_WIDTH),
                weights["decoder_attn_o_w"][i],
                buffers["decoder_x"],
                buffers["gate_buf"],
            )
            adarms_norm_style_proj_torch(
                buffers["decoder_x"],
                buffers["decoder_time_emb"][step],
                weights["decoder_pre_ffn_norm_mod_w"][i],
                weights["decoder_pre_ffn_norm_mod_b"][i],
                buffers["x_normed_buf"],
                buffers["gate_buf"],
                buffers["decoder_style_ffn"][step, i],
            )
            matmul_small_gate_torch(
                buffers["x_normed_buf"],
                weights["decoder_ffn_gate_w"][i],
                weights["decoder_ffn_up_w"][i],
                buffers["decoder_hidden"],
            )
            matmul_k_4096_1024_gate_torch(
                buffers["decoder_hidden"],
                weights["decoder_ffn_down_w"][i],
                buffers["decoder_x"],
                buffers["gate_buf"],
            )

        adarms_matmul_k_1024_32_bias_res_torch(
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


def pi05_model_torch(weights, buffers, num_views, encoder_seq_len, num_steps=NUM_STEPS):
    vision_encoder_torch(weights, buffers, num_views)
    transformer_encoder_torch(weights, buffers, encoder_seq_len)
    transformer_decoder_torch(weights, buffers, encoder_seq_len, num_steps)


class Pi05TorchInference:
    def __init__(
        self,
        checkpoint,
        num_views,
        chunk_size,
        tokenizer_path: str | None = None,
        max_tokenize_len: int = 200,
        discrete_state_input: bool = True,
        max_prompt_text: str | None = None,
        state_dim_for_max_prompt: int | None = None,
        max_prompt_len: int | None = None,
        compile_model: bool | None = None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("Pi05TorchInference requires CUDA")

        self.discrete_state_input = discrete_state_input
        self.tokenizer_path = tokenizer_path
        self.checkpoint = checkpoint
        self.num_views = num_views
        self.chunk_size = chunk_size
        self.max_tokenize_len = int(max_tokenize_len)
        if discrete_state_input:
            if not tokenizer_path or tokenizer_path == "fake-tokenizer":
                raise ValueError("Pi05TorchInference requires a real tokenizer_path when discrete_state_input=True")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if max_prompt_len is not None:
                self.max_prompt_len = int(max_prompt_len)
            elif max_prompt_text is not None and state_dim_for_max_prompt is not None:
                self.max_prompt_len = self.estimate_max_prompt_len(
                    tokenizer=self.tokenizer,
                    task_prompt=max_prompt_text,
                    state_dim=int(state_dim_for_max_prompt),
                    max_tokenize_len=self.max_tokenize_len,
                    state_token_value=255,
                )
            else:
                self.max_prompt_len = self.max_tokenize_len
        else:
            self.max_prompt_len = len(checkpoint["language_embeds"])
        print(f"torch max_prompt_len: {self.max_prompt_len}, max_tokenize_len: {self.max_tokenize_len}")

        self.weights = {
            "vision_patch_embedding_w": torch.empty(14, 14, 3, 1152, dtype=torch.bfloat16, device="cuda"),
            "vision_patch_embedding_b": torch.empty(1152, dtype=torch.bfloat16, device="cuda"),
            "vision_position_embedding": torch.empty(256, 1152, dtype=torch.bfloat16, device="cuda"),
            "vision_attn_qkv_w": torch.empty(27, 1152, 3 * 1152, dtype=torch.bfloat16, device="cuda"),
            "vision_attn_qkv_b": torch.empty(27, 3 * 1152, dtype=torch.bfloat16, device="cuda"),
            "vision_attn_o_w": torch.empty(27, 1152, 1152, dtype=torch.bfloat16, device="cuda"),
            "vision_attn_o_b": torch.empty(27, 1152, dtype=torch.bfloat16, device="cuda"),
            "vision_ffn_up_w": torch.empty(27, 1152, 4304, dtype=torch.bfloat16, device="cuda"),
            "vision_ffn_up_b": torch.empty(27, 4304, dtype=torch.bfloat16, device="cuda"),
            "vision_ffn_down_w": torch.empty(27, 4304, 1152, dtype=torch.bfloat16, device="cuda"),
            "vision_ffn_down_b": torch.empty(27, 1152, dtype=torch.bfloat16, device="cuda"),
            "vision_pre_attn_norm_w": torch.empty(27, 1152, dtype=torch.bfloat16, device="cuda"),
            "vision_pre_attn_norm_b": torch.empty(27, 1152, dtype=torch.bfloat16, device="cuda"),
            "vision_pre_ffn_norm_w": torch.empty(27, 1152, dtype=torch.bfloat16, device="cuda"),
            "vision_pre_ffn_norm_b": torch.empty(27, 1152, dtype=torch.bfloat16, device="cuda"),
            "vision_final_norm_w": torch.empty(1152, dtype=torch.bfloat16, device="cuda"),
            "vision_final_norm_b": torch.empty(1152, dtype=torch.bfloat16, device="cuda"),
            "encoder_multi_modal_projector_w": torch.empty(1152, 2048, dtype=torch.bfloat16, device="cuda"),
            "encoder_multi_modal_projector_b": torch.empty(2048, dtype=torch.bfloat16, device="cuda"),
            "encoder_attn_qkv_w": torch.empty(18, 2048, 2560, dtype=torch.bfloat16, device="cuda"),
            "encoder_attn_o_w": torch.empty(18, 2048, 2048, dtype=torch.bfloat16, device="cuda"),
            "encoder_ffn_gate_w": torch.empty(18, 2048, 16384, dtype=torch.bfloat16, device="cuda"),
            "encoder_ffn_up_w": torch.empty(18, 2048, 16384, dtype=torch.bfloat16, device="cuda"),
            "encoder_ffn_down_w": torch.empty(18, 16384, 2048, dtype=torch.bfloat16, device="cuda"),
            "decoder_time_embeds": torch.zeros(10, 1024, dtype=torch.bfloat16, device="cuda"),
            "decoder_time_mlp_in_w": torch.empty(1024, 1024, dtype=torch.bfloat16, device="cuda"),
            "decoder_time_mlp_in_b": torch.empty(1024, dtype=torch.bfloat16, device="cuda"),
            "decoder_time_mlp_out_w": torch.empty(1024, 1024, dtype=torch.bfloat16, device="cuda"),
            "decoder_time_mlp_out_b": torch.empty(1024, dtype=torch.bfloat16, device="cuda"),
            "decoder_action_in_proj_w": torch.empty(32, 1024, dtype=torch.bfloat16, device="cuda"),
            "decoder_action_in_proj_b": torch.empty(1024, dtype=torch.bfloat16, device="cuda"),
            "decoder_pre_attn_norm_mod_w": torch.empty(18, 1024, 3 * 1024, dtype=torch.bfloat16, device="cuda"),
            "decoder_pre_attn_norm_mod_b": torch.empty(18, 3 * 1024, dtype=torch.bfloat16, device="cuda"),
            "decoder_pre_ffn_norm_mod_w": torch.empty(18, 1024, 3 * 1024, dtype=torch.bfloat16, device="cuda"),
            "decoder_pre_ffn_norm_mod_b": torch.empty(18, 3 * 1024, dtype=torch.bfloat16, device="cuda"),
            "decoder_attn_qkv_w": torch.empty(18, 1024, 2560, dtype=torch.bfloat16, device="cuda"),
            "decoder_attn_o_w": torch.empty(18, 2048, 1024, dtype=torch.bfloat16, device="cuda"),
            "decoder_ffn_gate_w": torch.empty(18, 1024, 4096, dtype=torch.bfloat16, device="cuda"),
            "decoder_ffn_up_w": torch.empty(18, 1024, 4096, dtype=torch.bfloat16, device="cuda"),
            "decoder_ffn_down_w": torch.empty(18, 4096, 1024, dtype=torch.bfloat16, device="cuda"),
            "decoder_action_out_proj_w": torch.empty(1024, 32, dtype=torch.bfloat16, device="cuda"),
            "decoder_action_out_proj_b": torch.empty(32, dtype=torch.bfloat16, device="cuda"),
            "decoder_final_norm_mod_w": torch.empty(1024, 3 * 1024, dtype=torch.bfloat16, device="cuda"),
            "decoder_final_norm_mod_b": torch.empty(3 * 1024, dtype=torch.bfloat16, device="cuda"),
            "language_embeds": torch.empty(len(checkpoint["language_embeds"]), 2048, dtype=torch.bfloat16, device="cuda"),
        }

        encoder_seq_len = num_views * 256 + self.max_prompt_len
        decoder_seq_len = chunk_size
        self.buffers = {
            "observation_images_normalized": torch.empty(num_views, 224, 224, 3, dtype=torch.bfloat16, device="cuda"),
            "diffusion_noise": torch.empty(chunk_size, 32, dtype=torch.bfloat16, device="cuda"),
            "vision_x": torch.empty(num_views, 256, 1152, dtype=torch.bfloat16, device="cuda"),
            "vision_x_norm": torch.empty(num_views, 256, 1152, dtype=torch.bfloat16, device="cuda"),
            "vision_QKV": torch.empty(num_views, 256, 3 * 1152, dtype=torch.bfloat16, device="cuda"),
            "vision_hidden": torch.empty(num_views, 256, 4304, dtype=torch.bfloat16, device="cuda"),
            "vision_x_split_k_buf": torch.empty((num_views * 256 * 1152 * 4,), dtype=torch.float32, device="cuda"),
            "encoder_rope_weights": torch.empty(encoder_seq_len, 256, dtype=torch.bfloat16, device="cuda"),
            "encoder_x": torch.empty(encoder_seq_len, 2048, dtype=torch.bfloat16, device="cuda"),
            "encoder_x_norm": torch.empty(encoder_seq_len, 2048, dtype=torch.bfloat16, device="cuda"),
            "encoder_K": torch.empty(18, encoder_seq_len + decoder_seq_len, 256, dtype=torch.bfloat16, device="cuda"),
            "encoder_V": torch.empty(18, encoder_seq_len + decoder_seq_len, 256, dtype=torch.bfloat16, device="cuda"),
            "encoder_Q": torch.empty(encoder_seq_len * 8, 256, dtype=torch.bfloat16, device="cuda"),
            "encoder_hidden": torch.empty(encoder_seq_len, 16384, dtype=torch.bfloat16, device="cuda"),
            "valid_encoder_len": torch.empty((1,), dtype=torch.int32, device="cuda"),
            "encoder_logits_buf": torch.empty((encoder_seq_len * 8, encoder_seq_len), dtype=torch.float32, device="cuda"),
            "encoder_attn_buf": torch.empty((encoder_seq_len * 8, encoder_seq_len), dtype=torch.bfloat16, device="cuda"),
            "encoder_ctx_buf": torch.empty((encoder_seq_len * 8, 256), dtype=torch.bfloat16, device="cuda"),
            "decoder_rope_weights": torch.empty(decoder_seq_len, 256, dtype=torch.bfloat16, device="cuda"),
            "decoder_x": torch.empty((decoder_seq_len, 1024), dtype=torch.bfloat16, device="cuda"),
            "decoder_x_buf": torch.empty((decoder_seq_len, 1024), dtype=torch.bfloat16, device="cuda"),
            "decoder_action_buf": torch.empty((decoder_seq_len, 32), dtype=torch.bfloat16, device="cuda"),
            "decoder_time_emb": torch.empty((10, decoder_seq_len, 1024), dtype=torch.bfloat16, device="cuda"),
            "decoder_style_attn": torch.empty((10, 18, decoder_seq_len, 1024 * 3), dtype=torch.bfloat16, device="cuda"),
            "decoder_style_ffn": torch.empty((10, 18, decoder_seq_len, 1024 * 3), dtype=torch.bfloat16, device="cuda"),
            "decoder_style_final": torch.empty((10, decoder_seq_len, 1024 * 3), dtype=torch.bfloat16, device="cuda"),
            "decoder_norm_factor_buf": torch.empty((decoder_seq_len,), dtype=torch.bfloat16, device="cuda"),
            "decoder_q_buf": torch.empty((decoder_seq_len * 8, 256), dtype=torch.bfloat16, device="cuda"),
            "decoder_logits_buf": torch.empty((decoder_seq_len * 8, encoder_seq_len + decoder_seq_len), dtype=torch.float32, device="cuda"),
            "decoder_attn_buf": torch.empty((decoder_seq_len * 8, encoder_seq_len + decoder_seq_len), dtype=torch.bfloat16, device="cuda"),
            "decoder_hidden": torch.empty((decoder_seq_len, 4096), dtype=torch.bfloat16, device="cuda"),
            "decode_split_k_buf": torch.empty((2, decoder_seq_len, 1024), dtype=torch.float32, device="cuda"),
            "x_normed_buf": torch.empty((decoder_seq_len, 1024), dtype=torch.bfloat16, device="cuda"),
            "gate_buf": torch.empty((decoder_seq_len, 1024), dtype=torch.bfloat16, device="cuda"),
        }

        prefix_alloc = self.num_views * 256 + self.max_prompt_len
        max_pos = (self.num_views * 256 + self.max_prompt_len - 1) + self.chunk_size
        position_ids = torch.arange(max_pos + 1, device="cuda")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 256, 2, dtype=torch.float32, device="cuda") / 256))
        k_phase = inv_freq[None, :] * position_ids[:, None]
        k_cos = torch.cos(k_phase).to(torch.bfloat16)
        k_sin = torch.sin(k_phase).to(torch.bfloat16)
        self._rope_table = torch.cat([k_cos[:, :, None], k_sin[:, :, None]], 2).view(-1, 256)
        self.buffers["encoder_rope_weights"].copy_(self._rope_table[:prefix_alloc])

        self.buffers["valid_encoder_len"].fill_(self.num_views * 256 + 1)
        for key, value in checkpoint.items():
            if key != "embedding_weight":
                self.weights[key].copy_(_to_cuda_bf16(value))
        self.weights["decoder_action_out_proj_w"] *= -1.0 / NUM_STEPS
        self.weights["decoder_action_out_proj_b"] *= -1.0 / NUM_STEPS

        for step in range(NUM_STEPS):
            matmul_small_bias_silu_torch(
                self.weights["decoder_time_embeds"][step].view(1, -1),
                self.weights["decoder_time_mlp_in_w"],
                self.buffers["decoder_x_buf"],
                self.weights["decoder_time_mlp_in_b"],
            )
            self.buffers["decoder_x_buf"].copy_(
                self.buffers["decoder_x_buf"][:1].expand(decoder_seq_len, -1)
            )
            matmul_small_bias_silu_torch(
                self.buffers["decoder_x_buf"],
                self.weights["decoder_time_mlp_out_w"],
                self.buffers["decoder_time_emb"][step],
                self.weights["decoder_time_mlp_out_b"],
            )
            for i in range(NUM_LAYERS):
                matmul_small_bias_torch(
                    self.buffers["decoder_time_emb"][step],
                    self.weights["decoder_pre_attn_norm_mod_w"][i],
                    self.buffers["decoder_style_attn"][step, i],
                    self.weights["decoder_pre_attn_norm_mod_b"][i],
                )
                matmul_small_bias_torch(
                    self.buffers["decoder_time_emb"][step],
                    self.weights["decoder_pre_ffn_norm_mod_w"][i],
                    self.buffers["decoder_style_ffn"][step, i],
                    self.weights["decoder_pre_ffn_norm_mod_b"][i],
                )
            matmul_small_bias_torch(
                self.buffers["decoder_time_emb"][step],
                self.weights["decoder_final_norm_mod_w"],
                self.buffers["decoder_style_final"][step],
                self.weights["decoder_final_norm_mod_b"],
            )

        self.prompt_embedding = None
        self._prompt_embed_scale = None
        if self.discrete_state_input:
            if "embedding_weight" not in checkpoint:
                raise KeyError("checkpoint must contain 'embedding_weight' when discrete_state_input=True")
            emb_w_t = _to_cuda_bf16(checkpoint["embedding_weight"])
            self.prompt_embedding = nn.Embedding(
                num_embeddings=emb_w_t.shape[0],
                embedding_dim=emb_w_t.shape[1],
                device="cuda",
                dtype=torch.bfloat16,
            )
            with torch.no_grad():
                self.prompt_embedding.weight.copy_(emb_w_t)
            self._prompt_embed_scale = float(emb_w_t.shape[1] ** 0.5)
        self.encoder_seq_len = encoder_seq_len

        if compile_model is not None and not compile_model:
            global _COMPILE_ENABLED
            _COMPILE_ENABLED = False
        self.compile_model = _COMPILE_ENABLED
        if self.compile_model:
            print("torch.compile: enabled for torch kernel replacement blocks")

    def estimate_max_prompt_len(
        self,
        tokenizer: AutoTokenizer,
        task_prompt: str,
        state_dim: int,
        max_tokenize_len: int = 200,
        state_token_value: int = 255,
    ) -> int:
        task_prompt = task_prompt.strip().replace("_", " ")
        state_str = " ".join([str(int(state_token_value))] * int(state_dim))
        full_prompt = f"Task: {task_prompt}, State: {state_str};\nAction: "
        token_ids = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(max_tokenize_len),
            padding=False,
        )["input_ids"][0]
        return int(token_ids.shape[0])

    def build_prompt_embeds(
        self,
        task_prompt: str,
        state_tokens: np.ndarray,
    ) -> tuple[torch.Tensor, int]:
        task_prompt = task_prompt.strip().replace("_", " ")
        state_str = " ".join(map(str, state_tokens.tolist()))
        full_prompt = f"Task: {task_prompt}, State: {state_str};\nAction: "
        token_ids = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokenize_len,
            padding=False,
        )["input_ids"][0].to(device="cuda", non_blocking=True)
        embeds = self.prompt_embedding(token_ids) * self._prompt_embed_scale
        return embeds, int(embeds.shape[0])

    def get_decoder_rope_weights(self, prompt_len: int) -> torch.Tensor:
        start = self.num_views * 256 + prompt_len - 1
        end = start + self.chunk_size
        return self._rope_table[start:end]

    def record_run(self):
        pi05_model_torch(self.weights, self.buffers, self.num_views, self.encoder_seq_len)

    def forward(
        self,
        observation_images_normalized: torch.Tensor,
        diffusion_noise: torch.Tensor,
        task_prompt: str = None,
        state_tokens: np.ndarray = None,
    ) -> torch.Tensor:
        if self.discrete_state_input:
            prompt_embeds, prompt_len = self.build_prompt_embeds(
                task_prompt=task_prompt,
                state_tokens=state_tokens,
            )
        else:
            prompt_embeds = self.weights["language_embeds"]
            prompt_len = self.weights["language_embeds"].shape[0]
        if prompt_len > self.max_prompt_len:
            raise ValueError(
                f"prompt_len={prompt_len} exceeds allocated max_prompt_len={self.max_prompt_len}; "
                "initialize Pi05TorchInference with a larger max_prompt_len"
            )
        start = self.num_views * 256
        self.buffers["encoder_x"][start : start + prompt_len].copy_(prompt_embeds)
        self.buffers["valid_encoder_len"].fill_(start + prompt_len)
        self.buffers["decoder_rope_weights"].copy_(self.get_decoder_rope_weights(prompt_len))
        self.buffers["observation_images_normalized"].copy_(observation_images_normalized)
        self.buffers["diffusion_noise"].copy_(diffusion_noise)
        self.record_run()
        return self.buffers["diffusion_noise"]
