import argparse
import math
import pickle
from pathlib import Path

import torch
from safetensors import safe_open


VISION_PREFIX = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
LANG_PREFIX = "paligemma_with_expert.paligemma.model.language_model"
PROJECTOR_PREFIX = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
EXPERT_PREFIX = "paligemma_with_expert.gemma_expert.model"


EXPECTED_SHAPES = {
    "embedding_weight": (257152, 2048),
    "vision_patch_embedding_w": (14, 14, 3, 1152),
    "vision_patch_embedding_b": (1152,),
    "vision_position_embedding": (256, 1152),
    "vision_attn_qkv_w": (27, 1152, 3456),
    "vision_attn_qkv_b": (27, 3456),
    "vision_attn_o_w": (27, 1152, 1152),
    "vision_attn_o_b": (27, 1152),
    "vision_ffn_up_w": (27, 1152, 4304),
    "vision_ffn_up_b": (27, 4304),
    "vision_ffn_down_w": (27, 4304, 1152),
    "vision_ffn_down_b": (27, 1152),
    "vision_pre_attn_norm_w": (27, 1152),
    "vision_pre_attn_norm_b": (27, 1152),
    "vision_pre_ffn_norm_w": (27, 1152),
    "vision_pre_ffn_norm_b": (27, 1152),
    "vision_final_norm_w": (1152,),
    "vision_final_norm_b": (1152,),
    "encoder_multi_modal_projector_w": (1152, 2048),
    "encoder_multi_modal_projector_b": (2048,),
    "encoder_attn_qkv_w": (18, 2048, 2560),
    "encoder_attn_o_w": (18, 2048, 2048),
    "encoder_ffn_gate_w": (18, 2048, 16384),
    "encoder_ffn_up_w": (18, 2048, 16384),
    "encoder_ffn_down_w": (18, 16384, 2048),
    "decoder_time_embeds": (10, 1024),
    "decoder_time_mlp_in_w": (1024, 1024),
    "decoder_time_mlp_in_b": (1024,),
    "decoder_time_mlp_out_w": (1024, 1024),
    "decoder_time_mlp_out_b": (1024,),
    "decoder_action_in_proj_w": (32, 1024),
    "decoder_action_in_proj_b": (1024,),
    "decoder_pre_attn_norm_mod_w": (18, 1024, 3072),
    "decoder_pre_attn_norm_mod_b": (18, 3072),
    "decoder_pre_ffn_norm_mod_w": (18, 1024, 3072),
    "decoder_pre_ffn_norm_mod_b": (18, 3072),
    "decoder_attn_qkv_w": (18, 1024, 2560),
    "decoder_attn_o_w": (18, 2048, 1024),
    "decoder_ffn_gate_w": (18, 1024, 4096),
    "decoder_ffn_up_w": (18, 1024, 4096),
    "decoder_ffn_down_w": (18, 4096, 1024),
    "decoder_action_out_proj_w": (1024, 32),
    "decoder_action_out_proj_b": (32,),
    "decoder_final_norm_mod_w": (1024, 3072),
    "decoder_final_norm_mod_b": (3072,),
    "language_embeds": None,
}


def get_tensor(reader, name: str, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return reader.get_tensor(name).to(dtype=dtype)


def linear_weight(reader, name: str, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return reader.get_tensor(name).t().contiguous().to(dtype=dtype)


def pair_adjacent_rope_layout(weight: torch.Tensor, num_heads: int, head_dim: int = 256) -> torch.Tensor:
    """Convert Gemma half-split rotary layout to realtime-vla adjacent-pair layout."""
    in_dim = weight.shape[0]
    return (
        weight.reshape(in_dim, num_heads, 2, head_dim // 2)
        .permute(0, 1, 3, 2)
        .reshape(in_dim, num_heads * head_dim)
        .contiguous()
    )


def make_time_embeddings(num_steps: int = 10) -> torch.Tensor:
    dt = -1.0 / num_steps
    time = torch.tensor(1.0, dtype=torch.float32)
    min_period = 4e-3
    max_period = 4.0
    embedding_dim = 1024
    fraction = torch.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    embeddings = []
    for _ in range(num_steps):
        sinusoid_input = time.unsqueeze(-1) * (1.0 / period).unsqueeze(0) * 2 * math.pi
        embeddings.append(torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1))
        time = time + dt
    return torch.cat(embeddings, dim=0).to(torch.bfloat16)


def make_language_embeds(
    embedding_weight: torch.Tensor,
    prompt: str | None,
    tokenizer_path: str | None,
    max_length: int,
) -> torch.Tensor:
    if not prompt:
        return torch.zeros(1, embedding_weight.shape[1], dtype=torch.bfloat16)
    if not tokenizer_path:
        raise ValueError("--tokenizer_path is required when --prompt is set")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    prompt_text = [prompt.strip().replace("_", " ") + "\n"]
    input_ids = tokenizer(
        prompt_text,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )["input_ids"].squeeze(0)
    embeds = embedding_weight.index_select(0, input_ids.cpu()).to(torch.float32)
    embeds *= float(embedding_weight.shape[1] ** 0.5)
    return embeds.to(torch.bfloat16)


def validate_checkpoint(weights: dict[str, torch.Tensor]) -> None:
    expected_keys = set(EXPECTED_SHAPES)
    actual_keys = set(weights)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        raise ValueError(f"checkpoint key mismatch: missing={missing}, extra={extra}")

    for key, expected_shape in EXPECTED_SHAPES.items():
        tensor = weights[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{key} is {type(tensor)!r}, expected torch.Tensor")
        if expected_shape is not None and tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{key} shape {tuple(tensor.shape)} != {expected_shape}")
        if expected_shape is None and (tensor.ndim != 2 or tensor.shape[1] != 2048):
            raise ValueError(f"{key} shape {tuple(tensor.shape)} must be (prompt_len, 2048)")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{key} dtype {tensor.dtype} != torch.bfloat16")


def convert(model_path: Path, prompt: str | None, tokenizer_path: str | None, prompt_max_length: int) -> dict[str, torch.Tensor]:
    safetensors_path = model_path / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(safetensors_path)

    weights = {key: torch.empty(shape, dtype=torch.bfloat16) for key, shape in EXPECTED_SHAPES.items() if shape}

    with safe_open(str(safetensors_path), framework="pt", device="cpu") as reader:
        # Embeddings and action/time heads.
        weights["embedding_weight"].copy_(get_tensor(reader, "paligemma_with_expert.paligemma.lm_head.weight"))
        weights["language_embeds"] = make_language_embeds(
            weights["embedding_weight"],
            prompt=prompt,
            tokenizer_path=tokenizer_path,
            max_length=prompt_max_length,
        )

        weights["decoder_time_embeds"].copy_(make_time_embeddings(num_steps=10))
        weights["decoder_time_mlp_in_w"].copy_(linear_weight(reader, "time_mlp_in.weight"))
        weights["decoder_time_mlp_in_b"].copy_(get_tensor(reader, "time_mlp_in.bias"))
        weights["decoder_time_mlp_out_w"].copy_(linear_weight(reader, "time_mlp_out.weight"))
        weights["decoder_time_mlp_out_b"].copy_(get_tensor(reader, "time_mlp_out.bias"))
        weights["decoder_action_in_proj_w"].copy_(linear_weight(reader, "action_in_proj.weight"))
        weights["decoder_action_in_proj_b"].copy_(get_tensor(reader, "action_in_proj.bias"))
        weights["decoder_action_out_proj_w"].copy_(linear_weight(reader, "action_out_proj.weight"))
        weights["decoder_action_out_proj_b"].copy_(get_tensor(reader, "action_out_proj.bias"))

        # Vision tower.
        weights["vision_patch_embedding_w"].copy_(
            reader.get_tensor(f"{VISION_PREFIX}.embeddings.patch_embedding.weight")
            .permute(2, 3, 1, 0)
            .contiguous()
            .to(torch.bfloat16)
        )
        weights["vision_patch_embedding_b"].copy_(get_tensor(reader, f"{VISION_PREFIX}.embeddings.patch_embedding.bias"))
        weights["vision_position_embedding"].copy_(get_tensor(reader, f"{VISION_PREFIX}.embeddings.position_embedding.weight"))

        for i in range(27):
            prefix = f"{VISION_PREFIX}.encoder.layers.{i}"
            q = linear_weight(reader, f"{prefix}.self_attn.q_proj.weight")
            k = linear_weight(reader, f"{prefix}.self_attn.k_proj.weight")
            v = linear_weight(reader, f"{prefix}.self_attn.v_proj.weight")
            weights["vision_attn_qkv_w"][i].copy_(torch.cat([q, k, v], dim=1))
            weights["vision_attn_qkv_b"][i].copy_(
                torch.cat(
                    [
                        get_tensor(reader, f"{prefix}.self_attn.q_proj.bias"),
                        get_tensor(reader, f"{prefix}.self_attn.k_proj.bias"),
                        get_tensor(reader, f"{prefix}.self_attn.v_proj.bias"),
                    ],
                    dim=0,
                )
            )
            weights["vision_attn_o_w"][i].copy_(linear_weight(reader, f"{prefix}.self_attn.out_proj.weight"))
            weights["vision_attn_o_b"][i].copy_(get_tensor(reader, f"{prefix}.self_attn.out_proj.bias"))
            weights["vision_ffn_up_w"][i].copy_(linear_weight(reader, f"{prefix}.mlp.fc1.weight"))
            weights["vision_ffn_up_b"][i].copy_(get_tensor(reader, f"{prefix}.mlp.fc1.bias"))
            weights["vision_ffn_down_w"][i].copy_(linear_weight(reader, f"{prefix}.mlp.fc2.weight"))
            weights["vision_ffn_down_b"][i].copy_(get_tensor(reader, f"{prefix}.mlp.fc2.bias"))
            weights["vision_pre_attn_norm_w"][i].copy_(get_tensor(reader, f"{prefix}.layer_norm1.weight"))
            weights["vision_pre_attn_norm_b"][i].copy_(get_tensor(reader, f"{prefix}.layer_norm1.bias"))
            weights["vision_pre_ffn_norm_w"][i].copy_(get_tensor(reader, f"{prefix}.layer_norm2.weight"))
            weights["vision_pre_ffn_norm_b"][i].copy_(get_tensor(reader, f"{prefix}.layer_norm2.bias"))

        weights["vision_final_norm_w"].copy_(get_tensor(reader, f"{VISION_PREFIX}.post_layernorm.weight"))
        weights["vision_final_norm_b"].copy_(get_tensor(reader, f"{VISION_PREFIX}.post_layernorm.bias"))

        # PaliGemma language model encoder.
        weights["encoder_multi_modal_projector_w"].copy_(linear_weight(reader, f"{PROJECTOR_PREFIX}.weight"))
        weights["encoder_multi_modal_projector_b"].copy_(get_tensor(reader, f"{PROJECTOR_PREFIX}.bias"))

        for i in range(18):
            prefix = f"{LANG_PREFIX}.layers.{i}"
            attn_scale = 1.0 + reader.get_tensor(f"{prefix}.input_layernorm.weight").to(torch.float32)

            q = reader.get_tensor(f"{prefix}.self_attn.q_proj.weight").t().contiguous().to(torch.float32)
            k = reader.get_tensor(f"{prefix}.self_attn.k_proj.weight").t().contiguous().to(torch.float32)
            v = reader.get_tensor(f"{prefix}.self_attn.v_proj.weight").t().contiguous().to(torch.float32)
            q = pair_adjacent_rope_layout(q * attn_scale[:, None], num_heads=8)
            k = pair_adjacent_rope_layout(k * attn_scale[:, None], num_heads=1)
            v = v * attn_scale[:, None]
            weights["encoder_attn_qkv_w"][i].copy_(torch.cat([q, k, v], dim=1).to(torch.bfloat16))
            weights["encoder_attn_o_w"][i].copy_(linear_weight(reader, f"{prefix}.self_attn.o_proj.weight"))

            ffn_scale = 1.0 + reader.get_tensor(f"{prefix}.post_attention_layernorm.weight").to(torch.float32)
            gate = reader.get_tensor(f"{prefix}.mlp.gate_proj.weight").t().contiguous().to(torch.float32)
            up = reader.get_tensor(f"{prefix}.mlp.up_proj.weight").t().contiguous().to(torch.float32)
            weights["encoder_ffn_gate_w"][i].copy_((gate * ffn_scale[:, None]).to(torch.bfloat16))
            weights["encoder_ffn_up_w"][i].copy_((up * ffn_scale[:, None]).to(torch.bfloat16))
            weights["encoder_ffn_down_w"][i].copy_(linear_weight(reader, f"{prefix}.mlp.down_proj.weight"))

        # Action expert decoder.
        for i in range(18):
            prefix = f"{EXPERT_PREFIX}.layers.{i}"
            weights["decoder_pre_attn_norm_mod_w"][i].copy_(linear_weight(reader, f"{prefix}.input_layernorm.dense.weight"))
            weights["decoder_pre_attn_norm_mod_b"][i].copy_(get_tensor(reader, f"{prefix}.input_layernorm.dense.bias"))
            weights["decoder_pre_ffn_norm_mod_w"][i].copy_(linear_weight(reader, f"{prefix}.post_attention_layernorm.dense.weight"))
            weights["decoder_pre_ffn_norm_mod_b"][i].copy_(get_tensor(reader, f"{prefix}.post_attention_layernorm.dense.bias"))

            q = linear_weight(reader, f"{prefix}.self_attn.q_proj.weight").to(torch.float32)
            k = linear_weight(reader, f"{prefix}.self_attn.k_proj.weight").to(torch.float32)
            v = linear_weight(reader, f"{prefix}.self_attn.v_proj.weight").to(torch.float32)
            q = pair_adjacent_rope_layout(q, num_heads=8)
            k = pair_adjacent_rope_layout(k, num_heads=1)
            weights["decoder_attn_qkv_w"][i].copy_(torch.cat([q, k, v], dim=1).to(torch.bfloat16))
            weights["decoder_attn_o_w"][i].copy_(linear_weight(reader, f"{prefix}.self_attn.o_proj.weight"))
            weights["decoder_ffn_gate_w"][i].copy_(linear_weight(reader, f"{prefix}.mlp.gate_proj.weight"))
            weights["decoder_ffn_up_w"][i].copy_(linear_weight(reader, f"{prefix}.mlp.up_proj.weight"))
            weights["decoder_ffn_down_w"][i].copy_(linear_weight(reader, f"{prefix}.mlp.down_proj.weight"))

        weights["decoder_final_norm_mod_w"].copy_(linear_weight(reader, f"{EXPERT_PREFIX}.norm.dense.weight"))
        weights["decoder_final_norm_mod_b"].copy_(get_tensor(reader, f"{EXPERT_PREFIX}.norm.dense.bias"))

    validate_checkpoint(weights)
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LeRobot Pi05 safetensors to realtime-vla Pi05 pickle format.")
    parser.add_argument("--model_path", type=Path, required=True, help="Directory containing model.safetensors")
    parser.add_argument("--output", type=Path, required=True, help="Output converted_checkpoint.pkl path")
    parser.add_argument("--prompt", type=str, default=None, help="Optional fixed prompt for language_embeds")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer path/name used when --prompt is set")
    parser.add_argument("--prompt_max_length", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = convert(
        model_path=args.model_path,
        prompt=args.prompt,
        tokenizer_path=args.tokenizer_path,
        prompt_max_length=args.prompt_max_length,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)
    total_gib = sum(t.numel() * t.element_size() for t in weights.values()) / 1024**3
    print(f"Successfully converted {args.model_path} -> {args.output}")
    print(f"keys={len(weights)} total_tensor_size={total_gib:.2f} GiB")


if __name__ == "__main__":
    main()
