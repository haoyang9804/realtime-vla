import argparse
import gc
import importlib
import json
import pickle
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


IMPLEMENTATIONS = {
    "base": ("pi05_infer", "Pi05Inference"),
    "sdpa": ("pi05_infer_sdpa", "Pi05SdpaInference"),
    "flash": ("pi05_infer_flash", "Pi05FlashInference"),
}


class FakeTokenizer:
    def __call__(self, text, return_tensors="pt", truncation=True, max_length=200, padding=False):
        if isinstance(text, (list, tuple)):
            text = text[0]
        ids = [2]
        ids.extend((ord(ch) % 251) + 3 for ch in str(text)[: max_length - 1])
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def parse_args():
    parser = argparse.ArgumentParser(description="Verify realtime-vla Pi05 inference implementations.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("../models/pi05_base/converted_realtime_vla.pkl"),
        help="Converted realtime-vla Pi05 pickle checkpoint.",
    )
    parser.add_argument("--implementations", nargs="+", choices=IMPLEMENTATIONS.keys(), default=list(IMPLEMENTATIONS))
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="pick up the object")
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--discrete-state-input", action="store_true")
    parser.add_argument("--fake-tokenizer", action="store_true")
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def tensor_stats(tensor, original_dtype=None, original_device=None):
    tensor_f = tensor.float()
    stats = {
        "shape": list(tensor.shape),
        "dtype": str(original_dtype or tensor.dtype),
        "stats_dtype": str(tensor_f.dtype),
        "finite": bool(torch.isfinite(tensor_f).all().item()),
        "min": float(tensor_f.min().item()),
        "max": float(tensor_f.max().item()),
        "mean": float(tensor_f.mean().item()),
    }
    if original_device is not None:
        stats["device"] = str(original_device)
    return stats


def load_class(name):
    module_name, class_name = IMPLEMENTATIONS[name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_implementation(name, policy_cls, checkpoint, images, noise, args):
    cleanup_cuda()
    if args.discrete_state_input and args.fake_tokenizer:
        pi05_infer = importlib.import_module("pi05_infer")
        pi05_infer.AutoTokenizer.from_pretrained = lambda *_args, **_kwargs: FakeTokenizer()

    started = time.perf_counter()
    policy = policy_cls(
        checkpoint=checkpoint,
        num_views=args.num_views,
        chunk_size=args.chunk_size,
        tokenizer_path=args.tokenizer_path or ("fake-tokenizer" if args.fake_tokenizer else None),
        discrete_state_input=args.discrete_state_input,
        max_tokenize_len=200,
        max_prompt_text=args.prompt if args.discrete_state_input else None,
        state_dim_for_max_prompt=args.state_dim if args.discrete_state_input else None,
    )
    torch.cuda.synchronize()
    init_ms = (time.perf_counter() - started) * 1000.0

    state_tokens = np.arange(args.state_dim, dtype=np.int64) % 256
    with torch.inference_mode():
        if args.discrete_state_input:
            out = policy.forward(images, noise.clone(), args.prompt, state_tokens)
        else:
            out = policy.forward(images, noise.clone())
    torch.cuda.synchronize()

    out_dtype = out.dtype
    out_device = out.device
    out_cpu = out.detach().float().cpu()
    result = {
        "ok": True,
        "init_ms": init_ms,
        "output": tensor_stats(out_cpu, original_dtype=out_dtype, original_device=out_device),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }

    del policy
    del out
    cleanup_cuda()
    return out_cpu, result


def compare_to_base(outputs, results, args):
    base = outputs.get("base")
    if base is None:
        return
    for name, output in outputs.items():
        if name == "base":
            continue
        diff = (output - base).abs()
        base_abs = base.abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        max_rel = float((diff / base_abs.clamp_min(1e-5)).max().item())
        allclose = bool(torch.allclose(output, base, atol=args.atol, rtol=args.rtol))
        results[name]["compare_to_base"] = {
            "allclose": allclose,
            "atol": args.atol,
            "rtol": args.rtol,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "max_rel": max_rel,
        }
        results[name]["ok"] = bool(results[name]["ok"] and allclose)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for realtime-vla Pi05 implementation verification")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"checkpoint={args.checkpoint}")
    print(f"implementations={','.join(args.implementations)}")
    print(f"num_views={args.num_views} chunk_size={args.chunk_size} seed={args.seed}")
    print(f"discrete_state_input={args.discrete_state_input} fake_tokenizer={args.fake_tokenizer}")
    print(f"cuda_device={torch.cuda.get_device_name(0)}")

    with args.checkpoint.open("rb") as f:
        checkpoint = pickle.load(f)
    print(f"loaded_keys={len(checkpoint)}")

    images = torch.zeros(args.num_views, 224, 224, 3, dtype=torch.bfloat16, device="cuda")
    noise = torch.randn(args.chunk_size, 32, dtype=torch.bfloat16, device="cuda")

    outputs = {}
    results = {
        "checkpoint": str(args.checkpoint),
        "implementations": args.implementations,
        "num_views": args.num_views,
        "chunk_size": args.chunk_size,
        "seed": args.seed,
        "discrete_state_input": args.discrete_state_input,
        "fake_tokenizer": args.fake_tokenizer,
        "tokenizer_path": args.tokenizer_path,
        "cuda_device": torch.cuda.get_device_name(0),
        "results": {},
    }

    for name in args.implementations:
        print(f"running_impl={name}")
        try:
            policy_cls = load_class(name)
            output, impl_result = run_implementation(name, policy_cls, checkpoint, images, noise, args)
            outputs[name] = output
            results["results"][name] = impl_result
            print(
                f"impl={name} ok=True finite={impl_result['output']['finite']} "
                f"min={impl_result['output']['min']:.6f} max={impl_result['output']['max']:.6f} "
                f"mean={impl_result['output']['mean']:.6f} init_ms={impl_result['init_ms']:.2f}"
            )
        except Exception as exc:
            results["results"][name] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            print(f"impl={name} ok=False error_type={type(exc).__name__} error={exc}")

    compare_to_base(outputs, results["results"], args)
    for name in args.implementations:
        comparison = results["results"].get(name, {}).get("compare_to_base")
        if comparison:
            print(
                f"compare={name}_vs_base allclose={comparison['allclose']} "
                f"max_abs={comparison['max_abs']:.6f} mean_abs={comparison['mean_abs']:.6f} "
                f"max_rel={comparison['max_rel']:.6f} atol={args.atol} rtol={args.rtol}"
            )

    overall_ok = all(item.get("ok", False) and item.get("output", {}).get("finite", False) for item in results["results"].values())
    results["overall_ok"] = bool(overall_ok)

    output_json = args.output_json
    if output_json is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_json = Path("benchmark_results") / f"pi05_impl_verify_{stamp}.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote_json={output_json}")
    print(f"overall_ok={overall_ok}")

    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
