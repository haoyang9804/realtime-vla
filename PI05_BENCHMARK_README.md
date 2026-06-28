# Pi05 Benchmark

Run from the workspace:

```bash
cd /mlx_devbox/users/mahaoyang/playground
```

## Check Keepalive

Benchmark timing should be measured with GPU keepalive stopped.

Check whether keepalive is running:

```bash
ps -C python3 -o pid,ppid,stat,etime,cmd --no-headers | rg 'gpu_sm_keepalive.py' || true
```

Stop keepalive if needed:

```bash
python3 gpu_sm_guard.py stop-for-rollout
```

## Run Core Pi05 Benchmark

This uses the converted realtime-vla checkpoint's fixed `language_embeds` path and does not call a tokenizer.

Important: the current `../models/pi05_base/converted_realtime_vla.pkl` was converted without `--prompt`, so `language_embeds` is a generated zero placeholder with shape `[1, 2048]`. It is useful for exercising the non-tokenizer path, but it is not a real text prompt embedding. For prompt/state inference, use the real tokenizer flow below.

```bash
cd realtime-vla
python3 benchmark_pi05_realtime.py \
  --checkpoint ../models/pi05_base/converted_realtime_vla.pkl \
  --implementations base \
  --num-views 3 \
  --chunk-size 50 \
  --warmup 20 \
  --iterations 200 \
  --output-json benchmark_results/pi05_realtime_latest.json
```

View the result:

```bash
cat benchmark_results/pi05_realtime_latest.json
```

## Download The Real PaliGemma Tokenizer

The Pi05 checkpoint uses the PaliGemma/Gemma vocabulary (`embedding_weight` shape `[257152, 2048]`). For `discrete_state_input=True`, `pi05_infer.py` calls `AutoTokenizer.from_pretrained(tokenizer_path)` and uses token ids to index `embedding_weight`.

The expected tokenizer is `google/paligemma-3b-pt-224`. It is a gated Hugging Face repo, so the host must be authenticated with an account that has accepted access.

1. Request/accept access in a browser:

```text
https://huggingface.co/google/paligemma-3b-pt-224
```

2. Log in on this machine:

```bash
hf auth login
hf auth whoami
```

You can also use a token non-interactively:

```bash
export HF_TOKEN=<your_huggingface_token>
```

3. Download the tokenizer files:

```bash
cd /mlx_devbox/users/mahaoyang/playground/realtime-vla
mkdir -p ../models/tokenizers/google-paligemma-3b-pt-224
hf download google/paligemma-3b-pt-224 \
  --include 'tokenizer*' \
  --include 'special_tokens_map.json' \
  --local-dir ../models/tokenizers/google-paligemma-3b-pt-224
```

4. Check that `transformers` can load the local tokenizer:

```bash
python3 - <<'PY'
from transformers import AutoTokenizer

path = "../models/tokenizers/google-paligemma-3b-pt-224"
tokenizer = AutoTokenizer.from_pretrained(path)
print(type(tokenizer).__name__)
print("vocab_size", len(tokenizer))
PY
```

If the download command prints `Access denied. This repository requires approval.`, the machine is not authenticated with an approved Hugging Face account.

If `hf auth login` fails with `httpx.InvalidURL: Invalid port: ':'`, one of the proxy environment variables in that shell is malformed. Check them without printing your token:

```bash
env | grep -i proxy
python3 - <<'PY'
import httpx
httpx.Client().close()
print("httpx proxy config ok")
PY
```

Then retry login from a clean proxy environment, or fix the bad proxy value:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    hf auth login
```

On this devbox, the known-good proxy values are:

```bash
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
```

## Run With Real PaliGemma Tokenizer

Use this for the real `discrete_state_input=True` path. By default the benchmark builds 3 deterministic test cases. Each case has different image tensors, diffusion noise, state tokens, and language prompt. Before `Pi05Inference` is constructed, the script tokenizes every case and uses the maximum case prompt length as `max_prompt_len` for CUDA buffer/graph allocation.

The `flash` implementation uses official `flash_attn_varlen_func` for encoder MQA attention. Decoder attention still uses the local Triton flash kernel because its valid K/V set is `valid encoder prefix + decoder suffix`, which requires compaction before it can be represented as a standard varlen FlashAttention sequence.

The default per-case tensor shapes are:

- images: `[3, 224, 224, 3]`, BF16 CUDA
- state tokens: `[32]`, NumPy int64
- diffusion noise: `[50, 32]`, BF16 CUDA
- action output: `[50, 32]`, BF16 CUDA

```bash
python3 benchmark_pi05_realtime.py \
  --checkpoint ../models/pi05_base/converted_realtime_vla.pkl \
  --implementations base sdpa flash \
  --num-views 3 \
  --chunk-size 50 \
  --num-cases 3 \
  --discrete-state-input \
  --tokenizer-path ../models/tokenizers/google-paligemma-3b-pt-224 \
  --warmup 20 \
  --iterations 200 \
  --output-json benchmark_results/pi05_impl_benchmark_real_tokenizer_latest.json
```

Print the implementation comparison:

```bash
python3 - <<'PY'
import json

path = "benchmark_results/pi05_impl_benchmark_real_tokenizer_latest.json"
with open(path) as f:
    data = json.load(f)

base_p50 = data["results"]["base"]["latency_ms"]["p50"]
for name, result in data["results"].items():
    lat = result["latency_ms"]
    print(
        f"{name:5s} p50={lat['p50']:.3f}ms "
        f"mean={lat['mean']:.3f}ms p95={lat['p95']:.3f}ms "
        f"relative={lat['p50'] / base_p50:.3f}x"
    )
PY
```

Verify all Pi05 implementations with the same real tokenizer:

```bash
python3 verify_pi05_impls.py \
  --checkpoint ../models/pi05_base/converted_realtime_vla.pkl \
  --num-views 3 \
  --chunk-size 50 \
  --num-cases 3 \
  --discrete-state-input \
  --tokenizer-path ../models/tokenizers/google-paligemma-3b-pt-224 \
  --output-json benchmark_results/pi05_impl_verify_real_tokenizer_latest.json
```

To provide explicit language cases:

```bash
python3 verify_pi05_impls.py \
  --checkpoint ../models/pi05_base/converted_realtime_vla.pkl \
  --discrete-state-input \
  --tokenizer-path ../models/tokenizers/google-paligemma-3b-pt-224 \
  --case-prompts \
    "pick up the red cube" \
    "move the blue cup to the left side of the tray" \
    "open the drawer, place the object inside, then close the drawer"
```

## Fake Tokenizer Is Only For Structural Debugging

This does not validate real Pi05 tokenization. Use it only when debugging buffer shapes or control flow without Hugging Face access.

```bash
python3 benchmark_pi05_realtime.py \
  --checkpoint ../models/pi05_base/converted_realtime_vla.pkl \
  --discrete-state-input \
  --fake-tokenizer \
  --warmup 20 \
  --iterations 200
```

## Restore Keepalive

If you want to restore the 100% keepalive used in this workspace:

```bash
cd /mlx_devbox/users/mahaoyang/playground
setsid -f bash -c 'echo $$ > gpu_sm_keepalive.pid; exec python3 gpu_sm_keepalive.py --target 100 --matrix-size 6144 --duty-cycle 100 --period 0.2' > gpu_sm_keepalive.log 2>&1
```

Verify:

```bash
nvidia-smi dmon -s u -c 5
```
