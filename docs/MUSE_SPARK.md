# Run the experimental Muse Spark reference

<!-- Updated: 2026-10-28 -->

`strata-muse` executes Muse Spark's text graph from existing GGUF files, using
memory-mapped weights and bounded decoded row tiles. Build it explicitly; it is
**off by default** and does not change the normal [Qwen installation](MODELS.md).

Use its token-ID CLI for numerical checks, or the normal Terminal launch and
chat server with your own GGUF files and local engine build. Chat uses the GGUF's
own tokenizer and template, with Muse's text and reasoning framing. Muse has no
installer download, GUI model chooser entry or automatic engine update. Apps
can connect to the running server through its usual API.

This reference does not use Strata's Qwen expert cache, speculative decoder,
multi-GPU scheduler or image/audio encoders. Full-checkpoint correctness, text
quality and useful 816B performance need separate validation; a synthetic
fixture cannot establish them.

- [Build and check a synthetic model](#build-and-check-a-synthetic-model)
- [Launch text chat from local files](#launch-text-chat-from-local-files)
- [Opt-in quantized llama backend (Linux CPU/CUDA, multi-GPU)](MUSE_LLAMA.md)
- [Package a local engine](#package-a-local-engine)
- [Memory limits](#memory-limits)

## Build and check a synthetic model

Use the normal [build environment](DETAILS.md), with the Python packages in
[`requirements.txt`](../requirements.txt), including NumPy, and CMake/Ninja from
[`requirements-build.txt`](../requirements-build.txt) or your system packages.
No model download is needed for these commands. Run from the repository root:

```sh
cmake -S . -B build-muse \
  -DSTRATA_ENABLE_MUSE=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF
cmake --build build-muse --target strata-muse muse_model_test --parallel 2
ctest --test-dir build-muse -R '^muse_(model|protocol)_test$' --output-on-failure
python -m unittest discover -s tools/muse -p 'test_*.py'
python -m tools.muse.fixture --output-dir /tmp/muse-fixture
build-muse/src/muse/strata-muse \
  --model /tmp/muse-fixture/model.gguf \
  --tokens 1,5,8,3,7,2,9 --ctx 32 --steps 0 --backend cpu \
  --logits-file /tmp/muse-fixture/native-cpu.json
python -m tools.muse.compare \
  --reference /tmp/muse-fixture/reference.json \
  --actual /tmp/muse-fixture/native-cpu.json
```

GGML's row converters come from Strata's existing pinned dependency. Keep
`STRATA_NATIVE_EXPERTS=ON` (the default). An existing matching llama.cpp checkout
can be selected with `-DSTRATA_GGML_DIR=/absolute/path/to/llama.cpp` to avoid a
build-time fetch. This does not change the dependency pin.

## Check the CUDA adapter

Build another directory with `STRATA_ENABLE_CUDA=ON`. For an RTX 3060, CUDA
architecture 86 is appropriate:

```sh
cmake -S . -B build-muse-cuda \
  -DSTRATA_ENABLE_MUSE=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-muse-cuda --target strata-muse muse_model_test --parallel 2
ctest --test-dir build-muse-cuda -R '^muse_model_test$' --output-on-failure
build-muse-cuda/src/muse/muse_model_test --cuda
build-muse-cuda/src/muse/strata-muse \
  --model /tmp/muse-fixture/model.gguf \
  --tokens 1,5,8,3,7,2,9 --ctx 32 --steps 0 --backend cuda --gpu 0 \
  --logits-file /tmp/muse-fixture/native-cuda.json
python -m tools.muse.compare \
  --reference /tmp/muse-fixture/reference.json \
  --actual /tmp/muse-fixture/native-cuda.json
```

Select the architecture and device for your hardware. CUDA accelerates tiled
matrix-vector products through cuBLAS; attention, routing, normalization and row
dequantization remain on the CPU. This is not an optimized GPU inference engine.
This reference has no Muse HIP, SYCL or multi-GPU execution path. The separate
[quantized llama adapter](MUSE_LLAMA.md) supports an opt-in CUDA layer split.
With a HIP engine build, Muse uses its CPU path; the gfx906 compatibility mode
does not enable the NVIDIA adapter. Build Muse separately from the SYCL project.

The CTests above use the CPU backend. The explicit `muse_model_test --cuda`
check runs CUDA logits, four greedy steps, EOS/EOT stops and synthetic quantized
rows against CPU results. It uses logical device 0; set `CUDA_VISIBLE_DEVICES`
to a GPU UUID to check one physical card. The separate NumPy comparison checks
the full synthetic F32 logits against an independent implementation.

## Launch text chat from local files

Run from this checkout, with the Python runtime packages installed. The engine
is the separate `strata-muse` executable; the normal Qwen engine and its package
are left unchanged. Start with the synthetic chat fixture to check the wiring:

```sh
python -m tools.muse.chat_fixture --output-dir ./muse-chat-fixture
export STRATA_MUSE_EXE="$PWD/build-muse/src/muse/strata-muse"
export STRATA_MUSE_BACKEND=cpu
python strata.py launch --family muse \
  --model-path ./muse-chat-fixture/model.gguf --context 1024 --plan
python strata.py launch --family muse \
  --model-path ./muse-chat-fixture/model.gguf --context 1024 --yes --no-tui
```

The second launch starts the normal local server and prints its address. Use
the chat page or connect a client to `/v1/chat/completions` or `/v1/messages`.
`strata connect` prints client settings; its tool-call check is unsupported for
this text-only reference. `python strata.py launch --stop` stops the launch.

The chat fixture contains a complete synthetic byte tokenizer, a template and
weights designed to emit `synthetic Muse response`. It checks preparation,
native generation and API transport, not language-model quality. The numerical
fixture from `tools.muse.fixture` has `no_vocab` and cannot be launched as chat.

For your own model, replace `--model-path` with one GGUF, any named split shard,
or a folder containing exactly one model's shards. No model or tokenizer is
downloaded. The first shard must carry the GPT-2 byte-BPE tokenizer with
`pre=gpt-4o` or `llama4`, token types, BOS and EOS/EOT IDs, and a chat template.
Missing or inconsistent data is refused. Preparation exports the GGUF's own
tokenizer and template into the launch's state folder; it never substitutes a
Qwen template or changes the source GGUF files.

CPU is the default. To use a CUDA build for matvec tiles, select it explicitly:

```sh
export STRATA_MUSE_EXE="$PWD/build-muse-cuda/src/muse/strata-muse"
export STRATA_MUSE_BACKEND=cuda
python strata.py launch --family muse \
  --model-path ./muse-chat-fixture/model.gguf --context 1024 --gpu 0 --plan
```

Remove `--plan` and add `--no-tui` to start. `--gpu N` is a CUDA-visible device
ordinal. `CUDA_VISIBLE_DEVICES` can restrict or reorder that list, so it is not
necessarily the number printed by `nvidia-smi`. To select one physical card,
set `CUDA_VISIBLE_DEVICES` to its GPU UUID and use `--gpu 0`. The engine command
keeps that ordinal; it does not apply Qwen's layer split or GPU remapping.
With no explicit GPU and no `STRATA_MUSE_BACKEND`, launch uses the CPU.

The launch context defaults to `min(1024, GGUF context_length)`. `--context`
selects a different size within the trained context and the reference memory
budget. `--plan` reads the real shard headers and reports mapped bytes, FP32 KV
and scratch; it creates no tokenizer files and starts no engine. It does not
claim that the weights stay resident or predict a useful inference speed.

Chat parses Muse's user text, reasoning and commentary. ATEM tool
calls, pictures/audio/video, parallel slots, multiple GPUs, drafting, tuning,
KV format selection, logprobs, logit bias, prefix/checkpoint controls and Qwen
reasoning-budget recovery are unsupported. Unsupported options are refused
instead of being passed to another engine. Scaled RoPE and custom/interleaved
row encodings are also refused; see `strata_control/muse.py` for the launcher's
supported row types. A recognized encoding is not full-checkpoint validation.

The published Muse template supports `reasoning_effort` values `low`, `medium`,
`high` and `xhigh`; its default is `xhigh`. `high` and `xhigh` remain distinct.
It has no switch to disable reasoning. Requests for `minimal` or `none`, or
`chat_template_kwargs` with `enable_thinking: false`, return an error instead of
silently keeping reasoning enabled. Use `low` for the lowest supported effort.

### Native server protocol

`strata-muse --serve --model MODEL.gguf --max-context N --backend cpu|cuda`
uses the existing resident engine transport. It prints `INFO` and
`READY N stop`, accepts `GEN MAX_NEW [key=value ...] ID,ID,...`, streams `T ID`
and finishes with `DONE generated prompt prompt_ms decode_ms reason`.
`STOP` cancels the active request; `QUIT` or input EOF cancels it and exits.
The server adds `--serve` itself; saved engine arguments must not repeat it.

Requests are serialized. Each request resets the KV state and reads its whole
prompt; no prefix/session reuse or batching is advertised. The sampler accepts
temperature, top-k (capped at 64), top-p, min-p, seed and repeat/frequency/presence
penalties with a history window. A positive seed is reproducible within this
sampler; zero uses fresh entropy. Its RNG is independent of Qwen's GPU sampler.
Greedy reference mode remains available separately. Malformed commands return
`ERR`; unsupported image, batch, session and cache commands are refused.

## Package a local engine

You can keep the executable in `STRATA_MUSE_EXE`, or place a development package
under `engine-muse/` in this checkout, the selected Strata install or the chosen
data directory. A direct `engine-muse/strata-muse` also works. On Windows the
executable is `strata-muse.exe`. These are local files, not signed release-feed
artifacts.

For a Linux x86-64 CPU build, the existing
[engine packager](../packaging/ENGINE-COMPONENT.md) accepts Muse's entrypoint:

```sh
mkdir -p muse-stage
cp build-muse/src/muse/strata-muse muse-stage/
python packaging/package_engine.py --src muse-stage --backend cpu \
  --platform linux-x86_64 --version 0.0.0 --entrypoint strata-muse \
  --out ./engine-muse
python packaging/package_engine.py \
  --verify ./engine-muse/0.0.0-linux-x86_64-cpu
unset STRATA_MUSE_EXE STRATA_MUSE_BACKEND
python strata.py launch --family muse \
  --model-path ./muse-chat-fixture/model.gguf --plan
```

`0.0.0` names a development package. Package folders are immutable; choose a
new version or a new output folder for another build. Keep this package out of
the normal `components/engine/` directory: its entrypoint is Muse, so it must
not be selected as the Qwen engine. `STRATA_MUSE_EXE`, when set, takes precedence
over `engine-muse/` discovery.

The manifest records every packaged file's size and SHA-256. Local packages are
unsigned (`signature: null`); the launcher applies the existing signature policy
and full file verification when it finds a manifest. `--verify` also checks
imported library coverage. The launch config puts the executable's directory
on its library path. A CUDA package needs the matching CUDA runtime libraries
beside it and accurate architecture/driver records; a CPU package check does
not establish that CUDA bundle's completeness or hardware compatibility.

Offering Muse as an installed production model needs real artifact hashes,
package/driver checks on each offered target, and full-checkpoint, tokenizer,
template and end-to-end validation. No Muse release package or feed entry is
promised by this local integration.

## Supply your own GGUF

The serialized architecture must be `muse-spark`, with the tensor names and
converter-folded scales used by
[llama.cpp's Muse Spark graph](https://github.com/ggml-org/llama.cpp/blob/e09a78a25dec3f42175ad59ecc540a54bed0b9b2/src/models/muse-spark.cpp).
[Meta's model](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B)
and its conversions have their own license and access requirements.

Pass the first shard with `--model`; every sibling shard must be present and
unchanged throughout execution. The CLI never downloads files. Missing tensors,
invalid geometry, truncated shards, unsupported row encodings and scaled RoPE
produce errors rather than falling back to a Qwen graph. Matrix rows are decoded
with the pinned GGML converter; recognizing an encoding does not imply a native
Strata quantized kernel or full-checkpoint validation.

For numerical CLI mode, obtain token IDs with the checkpoint's tokenizer and
chat template. `--steps N` generates up to N greedy token IDs and stops at metadata
EOS/EOT. `--ctx` must hold the forced input plus requested generation. Output is
JSON with:

- `tokens`: the supplied forced input IDs;
- `logits`: one full next-token logit vector per forced input;
- `generated_tokens`: generated IDs, including an encountered EOS/EOT.

The numerical CLI's default context is 128 (`--ctx`, or `--max-context`, changes
it). Unknown/duplicate flags, invalid token IDs and
context overflow fail with a nonzero exit. Output must not alias any input
shard. Execution/write failures can leave incomplete JSON; check the exit code
before using it.

## Memory limits

Weights stay file-backed; no full model or expert layer is copied into a GPU
cache. KV buffers use a local-layer ring and bounded global-layer history. The
reference rejects calculated KV plus activation/tile scratch above 1 GiB.
Resident mode includes an additional 16 MiB command buffer and 16 bytes per
context token for protocol/history storage in that same 1 GiB budget. The launch
plan reports that allowance separately. The numerical CLI's budget is unchanged.
Headers, mapped-file/page-cache residency, allocator and library overhead are
**not** a total-process memory guarantee.

The CUDA adapter explicitly allocates two 4 MiB buffers and 256 output bytes.
CUDA context and cuBLAS overhead are additional, so available VRAM must exceed
those buffers. These limits do not make a large checkpoint fit the disk or
establish acceptable storage-backed inference speed. Coordinate resource use
before testing beside an existing model service.

## Understand the numerical check

The fixture contains locally generated, nonuniform F32 weights: hidden64,
head16, four query/two KV heads, latent16, FF32, four experts/top2, vocabulary64,
two layers, context32 and window3. Its local/RoPE and global/NoPE pattern is
`[true,false]`. Seven forced tokens cross the local-window boundary and yield
448 next-token logits.

The oracle is independently implemented in NumPy and reads back serialized F32
values. It covers embedding normalization, gated GQA, local masking/RoPE,
alpha/beta residuals, full-stream sigmoid routing with selection-only correction
bias, latent projections, per-expert input-scaled normalization before weighted
accumulation, final logit scaling and softcap. Norm deltas, Q/router scales and
residual gates are already folded in GGUF; they must not be folded twice.

Mutation tests verify that omitting or reordering these operations is detected.
The comparator requires matching tokens and full matrix shape, rejects nonfinite
or malformed values and exits 1 on disagreement. Its default elementwise bound
is `abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)`.

The fixture has no tokenizer, real model weights or multimodal inputs. It does
not validate a published quantized checkpoint, model quality, full context or
performance. Record those results separately in the issue or pull request.
