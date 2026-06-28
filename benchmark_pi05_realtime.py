import argparse
import gc
import importlib
import json
import pickle
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

import pi05_infer


IMPLEMENTATIONS = {
    "base": ("pi05_infer", "Pi05Inference"),
    "autotune": ("pi05_infer_autotune", "Pi05AutotuneInference"),
    "torch": ("pi05_infer_torch", "Pi05TorchInference"),
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


def latency_summary(values):
    return {
        "min": min(values),
        "p50": statistics.median(values),
        "mean": statistics.fmean(values),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


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


def load_class(name):
    module_name, class_name = IMPLEMENTATIONS[name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_policy_once(policy, case, args):
    if args.discrete_state_input:
        return policy.forward(case["images"], case["noise"], case["prompt"], case["state_tokens"])
    return policy.forward(case["images"], case["noise"])


def benchmark_implementation(name, policy_cls, checkpoint, cases, max_prompt_len, args):
    cleanup_cuda()
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    init_started = torch.cuda.Event(enable_timing=True)
    init_finished = torch.cuda.Event(enable_timing=True)

    init_started.record()
    policy = policy_cls(
        checkpoint=checkpoint,
        num_views=args.num_views,
        chunk_size=args.chunk_size,
        tokenizer_path=args.tokenizer_path or ("fake-tokenizer" if args.fake_tokenizer else None),
        discrete_state_input=args.discrete_state_input,
        max_tokenize_len=200,
        max_prompt_len=max_prompt_len,
    )
    init_finished.record()
    torch.cuda.synchronize()
    init_ms = float(init_started.elapsed_time(init_finished))

    with torch.inference_mode():
        for index in range(args.warmup):
            out = run_policy_once(policy, cases[index % len(cases)], args)
        torch.cuda.synchronize()

        timings_ms = []
        timings_by_case = {case["name"]: [] for case in cases}
        for index in range(args.iterations):
            case = cases[index % len(cases)]
            started.record()
            out = run_policy_once(policy, case, args)
            finished.record()
            torch.cuda.synchronize()
            elapsed_ms = float(started.elapsed_time(finished))
            timings_ms.append(elapsed_ms)
            timings_by_case[case["name"]].append(elapsed_ms)

    out_f = out.float()
    result = {
        "ok": True,
        "init_ms": init_ms,
        "output_shape": list(out.shape),
        "output_dtype": str(out.dtype),
        "output_device": str(out.device),
        "output_finite": bool(torch.isfinite(out_f).all().item()),
        "output_min": float(out_f.min().item()),
        "output_max": float(out_f.max().item()),
        "output_mean": float(out_f.mean().item()),
        "latency_ms": latency_summary(timings_ms),
        "latency_ms_by_case": {
            case_name: latency_summary(case_timings)
            for case_name, case_timings in timings_by_case.items()
            if case_timings
        },
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    del policy
    del out
    cleanup_cuda()
    return result


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
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--case-prompts", nargs="*", default=None)
    parser.add_argument("--implementations", nargs="+", choices=IMPLEMENTATIONS.keys(), default=list(IMPLEMENTATIONS))
    parser.add_argument("--discrete-state-input", action="store_true")
    parser.add_argument("--fake-tokenizer", action="store_true")
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    if args.discrete_state_input and not args.fake_tokenizer and not args.tokenizer_path:
        parser.error("--tokenizer-path is required when --discrete-state-input is set without --fake-tokenizer")
    if args.num_cases < 1:
        parser.error("--num-cases must be >= 1")
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for realtime-vla Pi05Inference")

    if args.discrete_state_input and args.fake_tokenizer:
        pi05_infer.AutoTokenizer.from_pretrained = lambda *_args, **_kwargs: FakeTokenizer()

    print(f"checkpoint={args.checkpoint}")
    print(
        f"num_views={args.num_views} chunk_size={args.chunk_size} "
        f"num_cases={args.num_cases} warmup={args.warmup} iterations={args.iterations}"
    )
    print(f"discrete_state_input={args.discrete_state_input}")
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
    implementation_results = {}
    for name in args.implementations:
        print(f"running_impl={name}")
        policy_cls = load_class(name)
        result = benchmark_implementation(name, policy_cls, checkpoint, cases, max_prompt_len, args)
        implementation_results[name] = result
        print(f"impl={name} output_shape={tuple(result['output_shape'])}")
        print(f"impl={name} output_dtype={result['output_dtype']}")
        print(f"impl={name} output_device={result['output_device']}")
        print(f"impl={name} output_finite={result['output_finite']}")
        for key, value in result["latency_ms"].items():
            print(f"impl={name} latency_{key}_ms={value:.4f}")

    stats = {
        "checkpoint": str(args.checkpoint),
        "implementations": args.implementations,
        "num_views": args.num_views,
        "chunk_size": args.chunk_size,
        "num_cases": len(cases),
        "cases": case_metadata,
        "max_prompt_len": max_prompt_len,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "discrete_state_input": args.discrete_state_input,
        "fake_tokenizer": args.fake_tokenizer,
        "tokenizer_path": args.tokenizer_path,
        "cuda_device": torch.cuda.get_device_name(0),
        "results": implementation_results,
    }

    output_json = args.output_json
    if output_json is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_json = Path("benchmark_results") / f"pi05_realtime_{stamp}.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote_json={output_json}")


if __name__ == "__main__":
    main()
