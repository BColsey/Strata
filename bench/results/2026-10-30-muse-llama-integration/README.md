# Muse UD-IQ1_S: three-GPU integration diagnostic

Measured on 2026-10-30. This checks the opt-in quantized backend and the existing
Strata server protocol with the full model. **It is not a standard benchmark,
normal-chat quality test or a demonstration of 50 tokens/s.**

The implementation is commit `ce32b8f3ba70e170c61548769bf7238c2c308fc8`.
See the [build and launch guide](../../../docs/MUSE_LLAMA.md). The reference
backend and Qwen defaults remain available and unchanged in numerical execution.

## Environment

- GPUs, in CUDA-visible order: RTX 2080 Ti 11 GiB, RTX 3080 10 GiB, RTX 3060
  12 GiB. PCIe widths: x4, x8, x4. This is a layer split, not tensor parallelism.
- Ryzen 9 3900X, about 64 GB installed RAM. Container limits: 12 CPU quota,
  48 GiB RAM, no swap, no network; model files mounted read-only.
- Linux x86_64; GNU 13.3.0, CUDA toolkit 13.0.48, Python 3.12.3. Built for
  `75;86`. The driver version, host distribution, storage device, power limits
  and clock settings were not captured with this probe.
- Source-built executable, not a release package. The binary hash, local
  container image ID and source hashes are in [results.json](results.json).
  The image ID is a local receipt, not a published image download.
- llama.cpp revision `e09a78a25dec3f42175ad59ecc540a54bed0b9b2`, in its own build
  tree; Strata's global GGML pin was not changed.
- Model: `unsloth/Muse-Spark-Open-816B-A42B-GGUF`, revision
  `c00670568b6ce5e56a247f67091247984df2a01e`, **UD-IQ1_S**, five GGUF shards,
  175,639,574,392 bytes total. This is not an FP32 checkpoint.
- Qwen was stopped; no other model benchmark was running. The GPUs were not
  clock-locked and OS/display activity was not eliminated.

## Method and settings

The retained [probe_real.py](probe_real.py) is the exact script used. It first
ran a separate CLI BOS-logit comparison, then started a resident executable via
`serve.server.StrataEngine`. It made three sequential requests in that process,
then cancelled another request and checked replay after cancellation.

Each measured request had prompt IDs `[200000]` (BOS only), temperature zero and
32 generated tokens. KV was cleared between requests, but expert and OS caches
could stay warm. The later requests repeated the identical prompt. No cache
flush, controlled cold-cache experiment, prefix reuse or speculation was used.

The resident engine arguments were:

```text
--serve
--model /model/Muse-Spark-Open-816B-A42B-UD-IQ1_S-00001-of-00005.gguf
--max-context 128 --backend llama-cuda --gpu 0,1,2
--moe-cache-mib 10240 --threads 12
```

The expert cache budget was **10 GiB total**, not per GPU. Dense weights, KV and
runtime buffers were additional. KV was F32, flash attention disabled, prefill
one token at a time. The backend allocated context 256; the advertised request
limit stayed at 128. Layer-offload log lines do **not** mean whole-model GPU
residency: expert weights remained CPU-backed with a selected-expert GPU cache.

The first generated token comes from prefill. Each successful 32-token request
therefore had **31 decode forwards**. Rates below are `31 * 1000 / decode_ms`;
they exclude startup and prefill. Cancelled requests are not throughput samples.

## Results

| Resident request | Prefill ms | Decode ms, 31 forwards | Decode forwards/s | Request wall s |
| --- | ---: | ---: | ---: | ---: |
| First request | 7458.88 | 46394.10 | 0.668188 | 53.858327 |
| Identical warm repeat 1 | 1733.21 | 15786.90 | 1.963653 | 17.520716 |
| Identical warm repeat 2 | 1175.09 | 15441.30 | 2.007603 | 16.616546 |

All three generated token arrays were identical. These are different cache
states, so they are shown individually rather than combined into one speed
claim. No matched before/after speedup was measured against the reference engine.
Resident startup took **182.747925 seconds**, separately from these requests.

Device-total `nvidia-smi` memory use just after READY was 9214, 9508 and 10200 MiB
in the order above. Those are startup snapshots, not peaks or process-only usage.
The [engine log excerpt](engine-log-excerpt.txt) records device selection,
allocated context and final aggregate cache counters. Those counters span the
whole resident session, including cancellation/replay, not one timed request.

The separate BOS check returned 202,048 finite logits. Every value matched the
retained independent CUDA output exactly (maximum absolute error zero). The
first two greedy IDs were `[15, 220]`. This is one-position numerical evidence,
not a claim of equivalence at all positions or full-model answer quality.
Cancellation emitted only `[15]`; the next request again produced `[15, 220]`.
The resident process exited with code zero.

## Validation

[Validation log excerpts](validation-log-excerpts.txt) retain passed checks,
baseline failures and corrected intermediate failures. Exit receipts, including
nonzero and missing statuses, are in `results.json`.

**Passed**

- CPU and CUDA builds; both builds passed their two CPU-execution CTests
  (11 protocol checks and four logit/option/safety checks).
- CUDA execution on all three selected cards: 11 synthetic protocol checks and
  two resident chat/server checks, including cancellation and request reset.
- Native reference model/protocol checks and byte-identical old/new CLI output
  on the numerical fixture. Reference CUDA F32, F16/Q8_0 and IQ1_S/Q6_K/Q8_0
  checks on each card. This does not establish byte identity on every workload.
- ROCm-toolchain gfx906 configuration: the CPU-only Muse reference built and
  passed model/protocol CTests. This was **not HIP GPU execution**.
- 24 `tools/muse` unit checks. Six actual local HTTP checks: chat completions,
  messages and responses, each streamed and non-streamed, using synthetic weights.
- Final precommit launcher/server check: 189 passed, one skipped, 326 subtests
  passed. The warning is retained in the log.

The final precommit command, from the repository root, was:

```sh
python3 -m pytest -q -p no:cacheprovider --maxfail=3 \
  tests/terminal/test_launch.py tests/terminal/test_launch_install.py \
  tests/terminal/test_launch_engine_first.py tests/terminal/test_launch_muse.py \
  tests/terminal/test_launch_dsv41.py tests/terminal/test_launch_identity.py \
  serve/test_muse.py
```

The engine source matches the implementation commit. The full-model probe ran
before the final launcher missing-executable diagnostic and test-only edits;
the final precommit check above covers those edits. The GPU probe was not rerun
for those changes.

**Failed or incomplete**

- The broader Terminal sweep stopped at two failures (378 passed, three skipped).
  Missing `gguf-py` and the Windows HIP PATH test both failed again on the untouched
  base `2e45edc5865288ed511d192bcfd8b1d92855fe77`. The remainder is not claimed passing.
  An earlier full-sweep attempt has no captured exit status; null is not success.
- Intermediate checks found a missing `vocab_size` in the synthetic `no_vocab`
  fixture and an empty callback invoked by the extracted reference CLI. Both
  were corrected before the final successful checks; their failures are retained.

**Not run / limits**

- Full-model normal-chat quality, varied prompts, long contexts, sustained decode
  and the standard benchmark protocol. The repeated BOS output is not useful-chat
  evidence. The six HTTP checks used a fixture, not the full checkpoint.
- No 50 tokens/s result or hardware guarantee. A larger cache or more RAM alone
  has not been shown to reach that target.
- No new-adapter HIP, SYCL, Windows or macOS validation; no signed packaging or
  production deployment. Models were preserved and Qwen was left stopped.

## Retained script and data

The probe ran as `python3 /out/probe_real.py` in a bounded container, with source
at `/src`, the build/evidence directory at `/out`, the checkpoint at `/model`
and earlier independent CUDA reference output at `/prior`. The wrapper set
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=12`, `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
selected the three physical GPU UUIDs in the order above. It imposed a 1200-second
timeout. The script refuses to overwrite `/out/real`.

This is a retained measurement harness, not a standalone download-and-run tool.
The model, binaries and prior `/prior/cuda-probe/logits/*.bin` FP32 reference
vector are not bundled. Replaying the full probe requires those inputs; it
asserts against that vector before measuring generation. Build instructions
are in the linked guide. Full raw logit vectors are also not bundled here.

The JSON preserves per-request timings, token IDs, cancellation and exit status.
GPU UUIDs, PCI addresses and random HIP fixture directory names were redacted in
the published data/logs. Numeric results were not changed. The script is unchanged;
its SHA-256 and the implementation source hashes are recorded in `results.json`.
