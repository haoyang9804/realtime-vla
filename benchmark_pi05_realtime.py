import argparse
import json
import pickle
import statistics
from datetime import datetime
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


def percentile(values, pct):
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark realtime-vla Pi05Inference with a converted checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("../models/pi05_base/converted_realtime_vla.pkl"),
        help="Converted realtime-vla Pi05 pickle checkpoint.",
    )
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="pick up the object")
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--discrete-state-input", action="store_true")
    parser.add_argument("--fake-tokenizer", action="store_true")
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for realtime-vla Pi05Inference")

    if args.discrete_state_input and args.fake_tokenizer:
        pi05_infer.AutoTokenizer.from_pretrained = lambda *_args, **_kwargs: FakeTokenizer()

    print(f"checkpoint={args.checkpoint}")
    print(f"num_views={args.num_views} chunk_size={args.chunk_size} warmup={args.warmup} iterations={args.iterations}")
    print(f"discrete_state_input={args.discrete_state_input}")
    print(f"cuda_device={torch.cuda.get_device_name(0)}")

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
    state_tokens = np.arange(args.state_dim, dtype=np.int64) % 256

    def run_once():
        if args.discrete_state_input:
            return policy.forward(images, noise, args.prompt, state_tokens)
        return policy.forward(images, noise)

    with torch.inference_mode():
        for _ in range(args.warmup):
            out = run_once()
        torch.cuda.synchronize()

        timings_ms = []
        for _ in range(args.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = run_once()
            end.record()
            torch.cuda.synchronize()
            timings_ms.append(float(start.elapsed_time(end)))

    out_f = out.float()
    stats = {
        "checkpoint": str(args.checkpoint),
        "num_views": args.num_views,
        "chunk_size": args.chunk_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "discrete_state_input": args.discrete_state_input,
        "fake_tokenizer": args.fake_tokenizer,
        "tokenizer_path": args.tokenizer_path,
        "cuda_device": torch.cuda.get_device_name(0),
        "output_shape": list(out.shape),
        "output_dtype": str(out.dtype),
        "output_device": str(out.device),
        "output_finite": bool(torch.isfinite(out_f).all().item()),
        "output_min": float(out_f.min().item()),
        "output_max": float(out_f.max().item()),
        "output_mean": float(out_f.mean().item()),
        "latency_ms": {
            "min": min(timings_ms),
            "p50": statistics.median(timings_ms),
            "mean": statistics.fmean(timings_ms),
            "p90": percentile(timings_ms, 90),
            "p95": percentile(timings_ms, 95),
            "p99": percentile(timings_ms, 99),
            "max": max(timings_ms),
            "stdev": statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        },
    }

    print("output_shape=" + str(tuple(out.shape)))
    print("output_dtype=" + str(out.dtype))
    print("output_device=" + str(out.device))
    print("output_finite=" + str(stats["output_finite"]))
    for key, value in stats["latency_ms"].items():
        print(f"latency_{key}_ms={value:.4f}")

    output_json = args.output_json
    if output_json is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_json = Path("benchmark_results") / f"pi05_realtime_{stamp}.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote_json={output_json}")


if __name__ == "__main__":
    main()
