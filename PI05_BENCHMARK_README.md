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

This uses the converted realtime-vla checkpoint and does not require the gated PaliGemma tokenizer.

```bash
cd realtime-vla
python3 benchmark_pi05_realtime.py \
  --checkpoint ../models/pi05_base/converted_realtime_vla.pkl \
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

## Run Discrete State Path With Fake Tokenizer

Use this to test the `discrete_state_input=True` structure without access to the gated tokenizer.

```bash
python3 benchmark_pi05_realtime.py \
  --checkpoint ../models/pi05_base/converted_realtime_vla.pkl \
  --discrete-state-input \
  --fake-tokenizer \
  --warmup 20 \
  --iterations 200
```

## Run With Real PaliGemma Tokenizer

`google/paligemma-3b-pt-224` is gated on Hugging Face. Use this only when you have a local tokenizer directory or authenticated access.

```bash
python3 benchmark_pi05_realtime.py \
  --checkpoint ../models/pi05_base/converted_realtime_vla.pkl \
  --discrete-state-input \
  --tokenizer-path /path/to/google/paligemma-3b-pt-224 \
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
