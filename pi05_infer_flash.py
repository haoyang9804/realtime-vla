from pi05_attention_flash import flash_mqa_decoder, flash_mqa_encoder
from pi05_infer import (
    Pi05Inference,
    adarms_matmul_k_1024_32_bias_res,
    adarms_norm_style_proj,
    layer_norm_matmul_n256_1152_2048_bias,
    matmul_k_1024_2560_qkv_rope,
    matmul_k_2048_1024_gate,
    matmul_k_32_1024_bias,
    matmul_k_4096_1024_gate,
    matmul_n_16384_2048_res,
    matmul_n_2048_2048_res,
    matmul_small_gate,
    rms_matmul_n_2048_16384_gate,
    rms_matmul_n_2048_2560_qkv_rope,
    vision_encoder,
)


def transformer_encoder_flash(weights, buffers, encoder_seq_len):
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
        rms_matmul_n_2048_2560_qkv_rope(
            buffers["encoder_x"],
            weights["encoder_attn_qkv_w"][i],
            buffers["encoder_rope_weights"],
            buffers["encoder_Q"],
            buffers["encoder_K"][i, :encoder_seq_len],
            buffers["encoder_V"][i, :encoder_seq_len],
            buffers["encoder_x_norm"],
        )
        if i != 17:
            flash_mqa_encoder(buffers, i, encoder_seq_len)
            matmul_n_2048_2048_res(
                buffers["encoder_ctx_buf"].view(-1, 2048),
                weights["encoder_attn_o_w"][i],
                buffers["encoder_x"],
            )
            rms_matmul_n_2048_16384_gate(
                buffers["encoder_x"],
                weights["encoder_ffn_gate_w"][i],
                weights["encoder_ffn_up_w"][i],
                buffers["encoder_hidden"],
                buffers["encoder_x_norm"],
            )
            matmul_n_16384_2048_res(
                buffers["encoder_hidden"],
                weights["encoder_ffn_down_w"][i],
                buffers["encoder_x"],
            )


def transformer_decoder_flash(weights, buffers, encoder_seq_len, num_steps=10):
    for step in range(num_steps):
        matmul_k_32_1024_bias(
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
            matmul_k_1024_2560_qkv_rope(
                buffers["x_normed_buf"],
                weights["decoder_attn_qkv_w"][i],
                buffers["decoder_rope_weights"],
                buffers["decoder_q_buf"],
                buffers["encoder_K"][i, encoder_seq_len : encoder_seq_len + seq_len],
                buffers["encoder_V"][i, encoder_seq_len : encoder_seq_len + seq_len],
            )
            flash_mqa_decoder(buffers, i, encoder_seq_len, seq_len)
            matmul_k_2048_1024_gate(
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
            matmul_small_gate[((seq_len + 127) // 128, (4096 + 63) // 64)](
                buffers["x_normed_buf"],
                weights["decoder_ffn_gate_w"][i],
                weights["decoder_ffn_up_w"][i],
                buffers["decoder_hidden"],
                seq_len,
                1024,
                4096,
            )
            matmul_k_4096_1024_gate(
                buffers["decoder_hidden"],
                weights["decoder_ffn_down_w"][i],
                buffers["decoder_x"],
                buffers["gate_buf"],
            )
        adarms_matmul_k_1024_32_bias_res(
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


def pi05_model_flash(weights, buffers, num_views, encoder_seq_len, num_steps=10):
    vision_encoder(weights, buffers, num_views)
    transformer_encoder_flash(weights, buffers, encoder_seq_len)
    transformer_decoder_flash(weights, buffers, encoder_seq_len, num_steps)


class Pi05FlashInference(Pi05Inference):
    def record_run(self):
        pi05_model_flash(self.weights, self.buffers, self.num_views, self.encoder_seq_len)
