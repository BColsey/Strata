# Run Muse with the quantized llama backend

<!-- Updated: 2026-10-30 -->

Use this **experimental, opt-in Linux CPU/CUDA backend** to keep weights quantized
and cache selected experts on one or more NVIDIA cards. It uses the existing
[Muse token-ID protocol and chat server](MUSE_SPARK.md#native-server-protocol).
It does not change the Qwen engine or replace the Muse numerical reference.
It does not promise 50 tokens per second.

See the [three-GPU integration diagnostic](../bench/results/2026-10-30-muse-llama-integration/README.md)
for recorded measurements, raw counters and untested cases. It is not a standard
benchmark or a full-model chat-quality test.

## Build and check on CPU

Run from this checkout. Use the build tools and Python packages described in
[the reference guide](MUSE_SPARK.md#build-and-check-a-synthetic-model).
The first configure downloads a checksum-verified llama.cpp source archive at
`e09a78a25dec3f42175ad59ecc540a54bed0b9b2`, not a model. It has its own build tree
and does not use or change `STRATA_GGML_DIR`.

```sh
cmake -S src/muse/llama -B build-muse-llama-cpu \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF
cmake --build build-muse-llama-cpu --target strata-muse-llama --parallel 2
ctest --test-dir build-muse-llama-cpu --output-on-failure
python -m tools.muse.chat_fixture --output-dir ./muse-chat-fixture
export STRATA_MUSE_EXE="$PWD/build-muse-llama-cpu/bin/strata-muse-llama"
export STRATA_MUSE_BACKEND=llama-cpu
python strata.py launch --family muse \
  --model-path ./muse-chat-fixture/model.gguf --context 1024 --plan
```

Remove `--plan` and add `--yes --no-tui` to start the local chat server. The
fixture tests wiring only. Replace its path with an existing Muse GGUF or split
model to use real weights. Tokenizer and template validation remain unchanged.

For an offline build, add `-DSTRATA_MUSE_LLAMA_ARCHIVE=/absolute/path/to/source.tar.gz`
with that revision's source archive. The same SHA-256 check still applies.
Keep the executable beside the matching libraries in the build's `bin/` folder;
copying only the executable is not a complete installation.

## Build for NVIDIA cards

Use a CUDA toolkit and driver supported by the pinned backend. This example
builds for an RTX 2080 Ti (75) and RTX 3080/3060 (86):

```sh
cmake -S src/muse/llama -B build-muse-llama-cuda \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  '-DCMAKE_CUDA_ARCHITECTURES=75;86'
cmake --build build-muse-llama-cuda --target strata-muse-llama --parallel 2
ctest --test-dir build-muse-llama-cuda --output-on-failure
python src/muse/test_protocol.py \
  --binary build-muse-llama-cuda/bin/strata-muse-llama \
  --backend llama-cuda --gpu 0,1,2 --moe-cache-mib 16
```

CTest uses CPU execution. The last command separately checks CUDA with synthetic
weights on all three selected cards. Choose devices and architectures for your
machine. A successful synthetic test is not a real-checkpoint quality check.

With other model services stopped and sufficient free VRAM, preview a launch:

```sh
export STRATA_MUSE_EXE="$PWD/build-muse-llama-cuda/bin/strata-muse-llama"
export STRATA_MUSE_BACKEND=llama-cuda
export STRATA_MUSE_MOE_CACHE_MIB=6144
export STRATA_MUSE_THREADS=12
python strata.py launch --family muse \
  --model-path ./muse-chat-fixture/model.gguf --context 1024 --gpu 0,1,2 --plan
```

Replace the fixture path with your own GGUF for a real launch. These are
CUDA-visible device ordinals, not necessarily `nvidia-smi` indices. Use
`CUDA_VISIBLE_DEVICES` with GPU UUIDs to select physical cards. The engine does
not apply Qwen GPU remapping. Missing devices fail rather than silently choosing
another card or CPU execution.

Layers are split equally across the chosen devices. The expert cache budget is
also divided equally; a nearly full smaller card can limit the whole setup.
This is a layer split, not a promise of simultaneous work on all cards.

## Options and memory

- `STRATA_MUSE_BACKEND`: `llama-cpu` or `llama-cuda`. The existing `cpu` and `cuda`
  values still select the reference executable.
- `STRATA_MUSE_MOE_CACHE_MIB`: total GPU expert-cache budget for `llama-cuda`;
  default 6144 MiB. Zero disables that cache. Dense weights, KV, CUDA contexts
  and scratch are **additional**. Start lower on small cards.
- `STRATA_MUSE_THREADS`: CPU thread count, 1–1024. If absent, use the pinned
  library's default. It does not reserve CPU cores.
- `--gpu`: a comma-separated list for `llama-cuda`; default device 0. CPU
  backends reject it. The reference CUDA backend still accepts only one card.

The launcher saves these as explicit engine arguments, so serving a saved config
does not depend on those environment variables remaining set. Expert-cache
options on a reference or CPU backend, and thread overrides on the reference,
are rejected rather than ignored.

`--plan` reads headers and starts no engine. It reports the requested expert
cache, not a fit guarantee. The quantized backend does **not** have the reference's
1 GiB live-host-memory guarantee. File mappings, OS page cache and runtime
allocations are additional. KV is F32 and the backend can round its allocated
context up; the advertised request limit still stays at the requested context.
Allocation failure is an error, not an automatic reduction in context or cache.

Large checkpoints can remain limited by disk/RAM transfers when experts miss the
GPU cache. A larger cache alone does not establish a target speed. Startup logs
show selected devices, model buffers, allocated context and cache size; shutdown
logs include the backend's cache counters. Logs go to stderr, never into the
token protocol.

## Check numerics and cancellation

The adapter also accepts the reference's `--tokens`, `--steps`, `--ctx` and
`--logits-file` arguments. It returns full forced-token logits and generated IDs.
The sampler, EOS/EOT handling, request reset and serialized `GEN`/`STOP`/`QUIT`
transport are shared with the reference. The numerical graph is not shared.

Prefill currently processes one token at a time. Each request clears KV; the
expert cache can stay warm across requests. There is no prefix reuse, batching,
speculative decoding or multimodal support. On CUDA, cancellation may wait for
the current forward pass to finish; its result is not emitted after cancellation.
The next request starts with cleared KV.

For completed, non-cancelled requests, the first generated token comes from
prefill. Count `generated_tokens - 1` decode forwards against `decode_ms`, rather
than dividing all output tokens by that time. Separate startup, prefill and
decode, and retain each run. A one-token response cannot measure decode speed.

## Return to the reference

Stop the launched server, then clear the opt-in settings:

```sh
python strata.py launch --stop
unset STRATA_MUSE_EXE STRATA_MUSE_BACKEND STRATA_MUSE_MOE_CACHE_MIB STRATA_MUSE_THREADS
```

Follow [the reference launch instructions](MUSE_SPARK.md#launch-text-chat-from-local-files)
to select its executable again. No model files are changed or removed. This
adapter has no signed release package or automatic update path. HIP, SYCL,
Windows and macOS builds are outside this experimental adapter's supported scope.
