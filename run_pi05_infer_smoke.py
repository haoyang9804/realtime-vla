import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

import pi05_infer
from pi05_infer import Pi05Inference


class FakeTokenizer:
    def __call__(self, text, return_tensors="pt", truncation=True, max_length=200, padding=False):
        if isinstance(text, (list, tuple)):
            text = text[0]
        ids = [2]
        ids.extend((ord(ch) % 251) + 3 for ch in str(text)[: max_length - 1])
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a minimal realtime-vla Pi05Inference CUDA smoke test.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("../models/pi05_base/converted_realtime_vla.pkl"),
        help="Converted realtime-vla Pi05 pickle checkpoint.",
    )
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="pick up the object")
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--discrete-state-input", action="store_true")
    parser.add_argument("--fake-tokenizer", action="store_true", help="Use a deterministic local tokenizer stub.")
    parser.add_argument("--tokenizer-path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.discrete_state_input and args.fake_tokenizer:
        pi05_infer.AutoTokenizer.from_pretrained = lambda *_args, **_kwargs: FakeTokenizer()

    print(f"loading_checkpoint={args.checkpoint}")
    with args.checkpoint.open("rb") as f:
        checkpoint = pickle.load(f)
    print(f"loaded_keys={len(checkpoint)}")

    policy = Pi05Inference(
        checkpoint=checkpoint,
        num_views=args.num_views,
        chunk_size=args.chunk_size,
        tokenizer_path=args.tokenizer_path or ("fake-tokenizer" if args.fake_tokenizer else None),
        discrete_state_input=args.discrete_state_input,
        max_tokenize_len=200,
        max_prompt_text=args.prompt if args.discrete_state_input else None,
        state_dim_for_max_prompt=args.state_dim if args.discrete_state_input else None,
    )
    print("policy_initialized=True")

    images = torch.zeros(args.num_views, 224, 224, 3, dtype=torch.bfloat16, device="cuda")
    noise = torch.randn(args.chunk_size, 32, dtype=torch.bfloat16, device="cuda")

    with torch.inference_mode():
        if args.discrete_state_input:
            state_tokens = np.arange(args.state_dim, dtype=np.int64) % 256
            out = policy.forward(images, noise, args.prompt, state_tokens)
        else:
            out = policy.forward(images, noise)
    torch.cuda.synchronize()

    out_f = out.float()
    print(f"output_shape={tuple(out.shape)}")
    print(f"output_dtype={out.dtype}")
    print(f"output_device={out.device}")
    print(f"output_finite={bool(torch.isfinite(out_f).all().item())}")
    print(f"output_min={float(out_f.min().item())}")
    print(f"output_max={float(out_f.max().item())}")
    print(f"output_mean={float(out_f.mean().item())}")


if __name__ == "__main__":
    main()
