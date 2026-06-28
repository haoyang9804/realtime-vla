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
from transformers import AutoTokenizer


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
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--case-prompts", nargs="*", default=None)
    parser.add_argument("--discrete-state-input", action="store_true")
    parser.add_argument("--fake-tokenizer", action="store_true")
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    if args.discrete_state_input and not args.fake_tokenizer and not args.tokenizer_path:
        parser.error("--tokenizer-path is required when --discrete-state-input is set without --fake-tokenizer")
    if args.num_cases < 1:
        parser.error("--num-cases must be >= 1")
    return args


def default_case_prompts(base_prompt, num_cases):
    prompts = [
        base_prompt,
        f"{base_prompt}; keep the wrist cameras aligned with the object",
        "move the red block to the left side of the tray and return to neutral",
        "open the drawer, place the object inside, close it, and keep the gripper clear",
    ]
    while len(prompts) < num_cases:
        prompts.append(f"{base_prompt}; repeat variation {len(prompts)} with a longer language instruction")
    return prompts[:num_cases]


def build_prompt_text(task_prompt, state_tokens):
    task_prompt = task_prompt.strip().replace("_", " ")
    state_str = " ".join(map(str, state_tokens.tolist()))
    return f"Task: {task_prompt}, State: {state_str};\nAction: "


def load_prompt_tokenizer(args):
    if not args.discrete_state_input:
        return None
    if args.fake_tokenizer:
        return FakeTokenizer()
    return AutoTokenizer.from_pretrained(args.tokenizer_path)


def prompt_token_len(tokenizer, prompt, state_tokens, max_tokenize_len):
    token_ids = tokenizer(
        build_prompt_text(prompt, state_tokens),
        return_tensors="pt",
        truncation=True,
        max_length=int(max_tokenize_len),
        padding=False,
    )["input_ids"][0]
    return int(token_ids.shape[0])


def make_cases(args):
    prompts = args.case_prompts or default_case_prompts(args.prompt, args.num_cases)
    cases = []
    for idx, prompt in enumerate(prompts):
        state_tokens = (np.arange(args.state_dim, dtype=np.int64) + idx * 17) % 256
        generator = torch.Generator(device="cuda")
        generator.manual_seed(args.seed + idx * 1009)
        images = torch.randn(
            args.num_views,
            224,
            224,
            3,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        ).to(torch.bfloat16)
        noise = torch.randn(
            args.chunk_size,
            32,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        ).to(torch.bfloat16)
        cases.append(
            {
                "name": f"case_{idx}",
                "prompt": prompt,
                "state_tokens": state_tokens,
                "images": images,
                "noise": noise,
            }
        )
    return cases


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


def run_implementation(name, policy_cls, checkpoint, cases, max_prompt_len, args):
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
        max_prompt_len=max_prompt_len,
    )
    torch.cuda.synchronize()
    init_ms = (time.perf_counter() - started) * 1000.0

    case_outputs = {}
    case_results = {}
    with torch.inference_mode():
        for case in cases:
            if args.discrete_state_input:
                out = policy.forward(
                    case["images"],
                    case["noise"].clone(),
                    case["prompt"],
                    case["state_tokens"],
                )
            else:
                out = policy.forward(case["images"], case["noise"].clone())
            torch.cuda.synchronize()
            out_dtype = out.dtype
            out_device = out.device
            out_cpu = out.detach().float().cpu()
            case_outputs[case["name"]] = out_cpu
            case_results[case["name"]] = {
                "output": tensor_stats(out_cpu, original_dtype=out_dtype, original_device=out_device)
            }

    result = {
        "ok": True,
        "init_ms": init_ms,
        "cases": case_results,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }

    del policy
    cleanup_cuda()
    return case_outputs, result


def compare_to_base(outputs, results, args):
    base_cases = outputs.get("base")
    if not base_cases:
        return
    for name, case_outputs in outputs.items():
        if name == "base":
            continue
        for case_name, output in case_outputs.items():
            base = base_cases[case_name]
            diff = (output - base).abs()
            base_abs = base.abs()
            max_abs = float(diff.max().item())
            mean_abs = float(diff.mean().item())
            max_rel = float((diff / base_abs.clamp_min(1e-5)).max().item())
            allclose = bool(torch.allclose(output, base, atol=args.atol, rtol=args.rtol))
            results[name]["cases"][case_name]["compare_to_base"] = {
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
    print(f"num_views={args.num_views} chunk_size={args.chunk_size} num_cases={args.num_cases} seed={args.seed}")
    print(f"discrete_state_input={args.discrete_state_input} fake_tokenizer={args.fake_tokenizer}")
    print(f"cuda_device={torch.cuda.get_device_name(0)}")

    with args.checkpoint.open("rb") as f:
        checkpoint = pickle.load(f)
    print(f"loaded_keys={len(checkpoint)}")

    cases = make_cases(args)
    prompt_lengths = {}
    max_prompt_len = None
    if args.discrete_state_input:
        prompt_tokenizer = load_prompt_tokenizer(args)
        prompt_lengths = {
            case["name"]: prompt_token_len(
                prompt_tokenizer,
                case["prompt"],
                case["state_tokens"],
                max_tokenize_len=200,
            )
            for case in cases
        }
        max_prompt_len = max(prompt_lengths.values())
        print(f"case_prompt_lengths={prompt_lengths}")
        print(f"allocated_max_prompt_len={max_prompt_len}")

    case_metadata = [
        {
            "name": case["name"],
            "prompt": case["prompt"],
            "state_dim": int(case["state_tokens"].shape[0]),
            "state_min": int(case["state_tokens"].min()),
            "state_max": int(case["state_tokens"].max()),
            "prompt_len": prompt_lengths.get(case["name"]),
            "image_shape": list(case["images"].shape),
            "noise_shape": list(case["noise"].shape),
        }
        for case in cases
    ]

    outputs = {}
    results = {
        "checkpoint": str(args.checkpoint),
        "implementations": args.implementations,
        "num_views": args.num_views,
        "chunk_size": args.chunk_size,
        "num_cases": len(cases),
        "cases": case_metadata,
        "max_prompt_len": max_prompt_len,
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
            case_outputs, impl_result = run_implementation(name, policy_cls, checkpoint, cases, max_prompt_len, args)
            outputs[name] = case_outputs
            results["results"][name] = impl_result
            case_finite = all(
                case_result["output"]["finite"]
                for case_result in impl_result["cases"].values()
            )
            print(
                f"impl={name} ok=True cases={len(impl_result['cases'])} "
                f"finite={case_finite} init_ms={impl_result['init_ms']:.2f}"
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
        for case_name, case_result in results["results"].get(name, {}).get("cases", {}).items():
            comparison = case_result.get("compare_to_base")
            if comparison:
                print(
                    f"compare={name}_{case_name}_vs_base allclose={comparison['allclose']} "
                    f"max_abs={comparison['max_abs']:.6f} mean_abs={comparison['mean_abs']:.6f} "
                    f"max_rel={comparison['max_rel']:.6f} atol={args.atol} rtol={args.rtol}"
                )

    overall_ok = all(
        item.get("ok", False)
        and all(case.get("output", {}).get("finite", False) for case in item.get("cases", {}).values())
        for item in results["results"].values()
    )
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
