"""Local-file preparation for the experimental Muse reference engine. No acquisition path."""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

from tools.gguf_reader import BLOCK_GEOMETRY, GGUFFile

ARCH = "muse-spark"
PREFIX = ARCH + "."
HOST_BUDGET = 1024 ** 3
TILE_BYTES = 4 * 1024 ** 2
COMMAND_BYTES = 16 * 1024 ** 2
SHARD = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.gguf$")
# Conservative subset of the pinned GGML row converters. Interleaved, integer,
# trellis and custom Q2_0 rows are deliberately not advertised by the launcher.
ROW_TYPES = {"F32", "F16", "BF16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
             "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS",
             "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS", "IQ1_M", "MXFP4"}


class MuseError(ValueError):
    pass


class _Header(GGUFFile):
    def _str(self, fh):
        (n,) = self._unpack(fh, "<Q")
        self._last_string = self._read(fh, n).decode("utf-8")
        return self._last_string

    def _value(self, fh):
        key = self._last_string
        if not hasattr(self, "_keys"):
            self._keys = set()
        if key in self._keys:
            raise MuseError("Duplicate Muse metadata key: " + key)
        self._keys.add(key)
        return super()._value(fh)

    def _read(self, fh, n):
        if n < 0 or self._pos + n > 256 * 1024 ** 2:
            raise MuseError("Muse GGUF header exceeds the 256 MiB reference limit")
        return super()._read(fh, n)

    def _count(self, fh, count, min_bytes, what):
        if what in {"metadata entries", "tensors"} and count > 100000:
            raise MuseError("Muse GGUF header has too many " + what)
        if what == "dimensions" and count > 4:
            raise MuseError("Muse tensors support at most four dimensions")
        if what.startswith("array") and count > 2000000:
            raise MuseError("Muse metadata array exceeds the reference limit")
        return super()._count(fh, count, min_bytes, what)


def _integer(value, name, zero=False):
    if type(value) is not int or not (0 if zero else 1) <= value <= 2 ** 31 - 1:
        raise MuseError("Muse requires valid integer metadata: " + name)
    return value


def _paths(path):
    path = Path(path).expanduser().absolute()
    if path.is_dir():
        files = sorted(path.glob("*.gguf"))
        groups = {SHARD.fullmatch(p.name).group(1) if SHARD.fullmatch(p.name) else p.name for p in files}
        if len(groups) != 1 or not files:
            raise MuseError("Choose one Muse GGUF or a folder containing exactly one model's shards")
        path = files[0]
    match = SHARD.fullmatch(path.name)
    if match:
        count = int(match[3])
        if not 1 <= int(match[2]) <= count <= 128:
            raise MuseError("Muse shard number/count is invalid (maximum 128 shards)")
        paths = tuple(path.with_name(f"{match[1]}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1))
    else:
        paths = (path,)
    for item in paths:
        if not item.is_file():
            raise MuseError("Missing Muse model shard: " + str(item))
    return paths


@dataclass
class Bundle:
    paths: tuple[Path, ...]
    metadata: dict
    geometry: dict
    context: int
    kv_bytes: int
    scratch_bytes: int
    mapped_bytes: int

    @property
    def protocol_bytes(self):
        return COMMAND_BYTES + self.context * 16

    def plan(self):
        return {"context": self.context, "trained_context": self.geometry["train_context"],
                "kv_bytes": self.kv_bytes, "scratch_bytes": self.scratch_bytes,
                "protocol_bytes": self.protocol_bytes,
                "host_live_bytes": self.kv_bytes + self.scratch_bytes + self.protocol_bytes, "host_budget_bytes": HOST_BUDGET,
                "mapped_bytes": self.mapped_bytes, "shards": len(self.paths),
                "lines": [f"Muse reference: {len(self.paths)} file(s), {self.mapped_bytes / 1e9:.3f} GB mapped weights; no resident expert cache.",
                          f"Context {self.context}: {(self.kv_bytes + self.scratch_bytes + self.protocol_bytes) / 2**20:.2f} MiB bounded host KV, scratch and protocol buffers (1 GiB limit).",
                          "Mapped weights, the OS page cache and runtime libraries are additional memory; full-checkpoint speed and memory use are unmeasured."],
                "warnings": []}


def inspect_model(path, context=None):
    """Check headers and payload bounds without loading weights or creating files."""
    paths = _paths(path)
    try:
        headers = [_Header(p) for p in paths]
    except (OSError, ValueError) as exc:
        raise MuseError(str(exc)) from None
    md = headers[0].metadata
    if md.get("general.architecture") != ARCH:
        raise MuseError("These files are not a Muse Spark GGUF (architecture muse-spark)")
    total = md.get("split.tensors.count")
    tensors = {}
    for i, header in enumerate(headers):
        hm = header.metadata
        if header.alignment > 1024 ** 2 or header.alignment & (header.alignment - 1):
            raise MuseError("Muse GGUF alignment must be a power of two within 1 MiB")
        if hm.get("general.architecture", ARCH) != ARCH:
            raise MuseError("Muse shards disagree about architecture")
        if len(paths) > 1 and (hm.get("split.count") != len(paths) or hm.get("split.no") != i or
                               (total is not None and hm.get("split.tensors.count") != total)):
            raise MuseError("Muse shard split.count/split.no/split.tensors.count does not match its file name")
        if len(paths) == 1 and hm.get("split.count", 1) != 1:
            raise MuseError("Muse split model needs every shard with its original split file names")
        ranges = []
        for tensor in header.tensors:
            if tensor.name in tensors:
                raise MuseError("Duplicate Muse tensor: " + tensor.name)
            if tensor.type_name not in ROW_TYPES:
                raise MuseError(f"Muse launcher does not support tensor encoding {tensor.type_name}: {tensor.name}")
            if not 1 <= len(tensor.shape) <= 4:
                raise MuseError("Invalid Muse tensor dimensions: " + tensor.name)
            for value in tensor.shape:
                _integer(value, tensor.name)
            block, size = BLOCK_GEOMETRY[tensor.type_name]
            if tensor.shape[0] % block:
                raise MuseError("Invalid Muse quantized row size: " + tensor.name)
            end = tensor.offset + math.prod(tensor.shape) // block * size
            if tensor.offset % header.alignment or end > header.path.stat().st_size - header.data_start:
                raise MuseError("Muse tensor exceeds its shard payload: " + tensor.name)
            ranges.append((tensor.offset, end))
            tensors[tensor.name] = tensor
        ranges.sort()
        if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
            raise MuseError("Overlapping Muse tensor payloads")
    if total is not None and total != len(tensors):
        raise MuseError("Muse split.tensors.count does not match the tensor directories")

    def n(key):
        return _integer(md.get(PREFIX + key), key)

    g = {"hidden": n("embedding_length"), "layers": n("block_count"), "heads": n("attention.head_count"),
         "kv_heads": n("attention.head_count_kv"), "head_dim": n("attention.key_length"),
         "latent": n("moe_latent_size"), "experts": n("expert_count"), "top_k": n("expert_used_count"),
         "train_context": n("context_length"), "window": n("attention.sliding_window")}
    if n("attention.value_length") != g["head_dim"] or g["heads"] % g["kv_heads"] or g["top_k"] > g["experts"]:
        raise MuseError("Muse GQA/head/expert geometry is incompatible")
    rotary = _integer(md.get(PREFIX + "rope.dimension_count", g["head_dim"]), "rope.dimension_count")
    if rotary > g["head_dim"] or rotary % 2:
        raise MuseError("Muse needs even rotary dimensions within the attention head")
    if md.get(PREFIX + "rope.scaling.type", "none") != "none" or md.get(PREFIX + "rope.scaling.factor", 1.0) != 1.0:
        raise MuseError("Scaled RoPE is not supported by the Muse reference")
    if n("expert_gating_func") != 2 or md.get(PREFIX + "expert_weights_norm") is not True:
        raise MuseError("Muse reference requires normalized sigmoid expert routing")
    pattern = md.get(PREFIX + "attention.sliding_window_pattern")
    if not isinstance(pattern, list) or len(pattern) != g["layers"] or any(type(x) is not bool for x in pattern):
        raise MuseError("Muse sliding-window pattern must contain one boolean per layer")
    ff = md.get(PREFIX + "expert_feed_forward_length")
    ff = ff if isinstance(ff, list) else [ff] * g["layers"]
    if len(ff) != g["layers"]:
        raise MuseError("Muse expert FF dimensions must match the layer count")
    ff = [_integer(x, "expert_feed_forward_length") for x in ff]
    for key in ("attention.layer_norm_rms_epsilon", "post_norm_epsilon", "rope.freq_base_swa", "logit_scale", "expert_weights_scale"):
        value = md.get(PREFIX + key)
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise MuseError("Muse requires positive finite metadata: " + key)
    softcap = md.get(PREFIX + "final_logit_softcapping")
    if type(softcap) not in (float, int) or not math.isfinite(softcap) or softcap < 0:
        raise MuseError("Muse logit softcap must be finite and nonnegative")

    def shape(name, dims):
        if name not in tensors or tensors[name].shape != dims:
            raise MuseError("Missing or malformed Muse tensor: " + name)

    emb = tensors.get("token_embd.weight")
    if emb is None or len(emb.shape) != 2:
        raise MuseError("Missing Muse token embedding")
    g["vocab"] = emb.shape[1]
    shape("token_embd.weight", [g["hidden"], g["vocab"]])
    shape("output.weight", [g["hidden"], g["vocab"]])
    shape("output_norm.weight", [g["hidden"]])
    q, kv = g["heads"] * g["head_dim"], g["kv_heads"] * g["head_dim"]
    for layer in range(g["layers"]):
        p = f"blk.{layer}."
        for name in ("attn_norm", "attn_resid_alpha", "attn_resid_beta", "ffn_norm", "post_ffw_norm", "ffn_resid_alpha", "ffn_resid_beta"):
            shape(p + name + ".weight", [g["hidden"]])
        for name in ("attn_q_norm", "attn_k_norm"):
            shape(p + name + ".weight", [g["head_dim"]])
        for name, dims in {"attn_q": [g["hidden"], q], "attn_k": [g["hidden"], kv], "attn_v": [g["hidden"], kv],
                           "attn_gate": [g["hidden"], q], "attn_output": [q, g["hidden"]],
                           "ffn_gate_inp": [g["hidden"], g["experts"]], "ffn_latent_down": [g["hidden"], g["latent"]],
                           "ffn_latent_up": [g["latent"], g["hidden"]], "ffn_pre_exp_norm": [g["latent"]],
                           "ffn_post_exp_norm": [g["latent"]], "ffn_gate_exps": [g["latent"], ff[layer], g["experts"]],
                           "ffn_up_exps": [g["latent"], ff[layer], g["experts"]],
                           "ffn_down_exps": [ff[layer], g["latent"], g["experts"]]}.items():
            shape(p + name + ".weight", dims)
        shape(p + "exp_probs_b.bias", [g["experts"]])
    width = max(g["hidden"], g["vocab"], q, kv, g["latent"], g["experts"], *ff)
    if width > TILE_BYTES // 4:
        raise MuseError("Muse geometry exceeds the bounded row/activation size")
    ctx = min(1024, g["train_context"]) if context is None else _integer(context, "context")
    if ctx > g["train_context"]:
        raise MuseError(f"Muse context {ctx} exceeds the GGUF trained context {g['train_context']}; RoPE scaling is unsupported")
    kv_bytes = sum(min(ctx, g["window"]) if local else ctx for local in pattern) * kv * 2 * 4
    scratch = TILE_BYTES + (width * 64 + ctx) * 4
    if kv_bytes + scratch + COMMAND_BYTES + ctx * 16 > HOST_BUDGET:
        raise MuseError("Muse KV, scratch and protocol buffers exceed the 1 GiB resident budget; choose a smaller --context")
    bundle = Bundle(paths, md, g, ctx, kv_bytes, scratch, sum(p.stat().st_size for p in paths))
    _tokenizer(bundle)
    return bundle


def _tokenizer(bundle):
    md, vocab = bundle.metadata, bundle.geometry["vocab"]
    tokens, merges, types = (md.get("tokenizer.ggml." + k) for k in ("tokens", "merges", "token_type"))
    if md.get("tokenizer.ggml.model") != "gpt2" or md.get("tokenizer.ggml.pre") not in {"gpt-4o", "llama4"}:
        raise MuseError("Muse launch requires its GGUF GPT-2 BPE tokenizer with pre=gpt-4o or llama4")
    if not isinstance(tokens, list) or len(tokens) != vocab or any(not isinstance(x, str) for x in tokens) or len(set(tokens)) != vocab:
        raise MuseError("Muse GGUF tokenizer vocabulary is missing or does not match the embedding")
    if not isinstance(merges, list) or any(not isinstance(x, str) for x in merges):
        raise MuseError("Muse GGUF tokenizer merges are missing")
    known = set(tokens)
    if len(set(merges)) != len(merges) or any(len(x.split(" ")) != 2 or any(p not in known for p in x.split(" ")) for x in merges):
        raise MuseError("Muse GGUF tokenizer has invalid or duplicate BPE merges")
    if not isinstance(types, list) or len(types) != vocab or any(type(x) is not int or not 1 <= x <= 6 for x in types):
        raise MuseError("Muse GGUF tokenizer token types are missing or invalid")
    special = {}
    for key, value in md.items():
        if key.startswith("tokenizer.ggml.") and key.endswith("_token_id"):
            value = _integer(value, key, zero=True)
            if value >= vocab:
                raise MuseError("Muse special token is outside the vocabulary: " + key)
            special[key.removeprefix("tokenizer.ggml.").removesuffix("_token_id")] = value
    if "bos" not in special or not any(k in special for k in ("eos", "eot")):
        raise MuseError("Muse GGUF must declare a BOS and an EOS or EOT token")
    template = md.get("tokenizer.chat_template")
    if not isinstance(template, str) or not template.strip():
        raise MuseError("Muse GGUF chat template is missing; no Qwen template is substituted")
    return tokens, merges, types, special, template


def prepare_tokenizer(bundle, out_dir):
    """Export the GGUF's own tokenizer and template into the supplied sidecar folder."""
    tokens, merges, types, special, template = _tokenizer(bundle)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"model": "gpt2", "pre": bundle.metadata["tokenizer.ggml.pre"], "vocab_size": len(tokens),
           "n_merges": len(merges), "special_ids": special, "add_bos_token": False}
    for name, value in (("vocab.json", {t: i for i, t in enumerate(tokens)}), ("token_type.json", types), ("tokenizer.json", cfg)):
        (out / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (out / "chat_template.jinja").write_text(template, encoding="utf-8")
    return out


def find_engine(roots):
    own = os.environ.get("STRATA_MUSE_EXE", "").strip()
    choices = [Path(own).expanduser()] if own else []
    if not own:
        for root in roots:
            folder = Path(root) / "engine-muse"
            for name in ("strata-muse", "strata-muse.exe"):
                choices.append(folder / name)
                choices.extend(sorted(folder.glob("*/" + name), reverse=True))
    for path in choices:
        if path.is_file() and (os.name == "nt" or os.access(path, os.X_OK)):
            manifest = path.parent / "engine-component.json"
            if manifest.is_file():
                from strata_control import engine_component as ec
                try:
                    component = ec.read_manifest(path.parent)
                    if component.exe != path:
                        raise MuseError("Muse package entrypoint is not strata-muse")
                    ec.verify_signature(component)
                    ec.verify(component, full=True)
                except ec.ComponentProblem as exc:
                    raise MuseError("Local Muse engine package failed its integrity check: " + str(exc)) from None
            return path.absolute()
    raise MuseError("Muse needs a local strata-muse executable: set STRATA_MUSE_EXE or put it in engine-muse/. No engine is downloaded")


def validate_config(cfg):
    """Validate a local Muse server config before a process is started."""
    args = cfg.get("args") or []
    allowed = {"--model", "--max-context", "--backend", "--gpu"}
    if not isinstance(args, list) or any(not isinstance(a, str) for a in args) or len(args) % 2 or any(flag not in allowed for flag in args[::2]) or len(set(args[::2])) != len(args[::2]):
        raise MuseError("Muse config supports only --model, --max-context, --backend and --gpu, each once")
    if cfg.get("gpu") is not None or cfg.get("parallel", 1) not in (None, 1) or cfg.get("drafting") not in (None, "off") or any(cfg.get(k) for k in
            ("vision", "replicas", "layer_split", "gpu_roles", "gpu_order", "gpu", "gpu_none",
             "thinking_budget_tokens", "reasoning_budget_tokens", "reasoning_effort_at_end", "effort_position",
             "reasoning_close_retry", "reasoning_loop_recovery", "tool_call_recovery", "user_args", "split_serve",
             "slot_save_path", "images", "codex_compaction_cache", "codex_thread_titles")):
        raise MuseError("Muse reference does not support pictures, parallel/replica slots, GPU splitting, drafting or reasoning-budget recovery")
    def arg(flag, default=None):
        if flag not in args:
            return default
        i = args.index(flag)
        if i + 1 >= len(args):
            raise MuseError("Muse engine option needs a value: " + flag)
        return args[i + 1]
    if arg("--backend", "cpu") not in {"cpu", "cuda"}:
        raise MuseError("Muse supports CPU or CUDA only")
    if cfg.get("backend", arg("--backend", "cpu")) != arg("--backend", "cpu"):
        raise MuseError("Muse config backend must match its engine arguments")
    if arg("--gpu") is not None:
        try:
            _integer(int(arg("--gpu")), "GPU index", zero=True)
        except (ValueError, TypeError):
            raise MuseError("Muse GPU index must be a nonnegative integer") from None
        if arg("--backend", "cpu") != "cuda":
            raise MuseError("Muse --gpu requires the CUDA backend")
    model = arg("--model")
    if not model:
        raise MuseError("Muse config must name --model GGUF")
    try:
        context = int(arg("--max-context", 128))
    except (ValueError, TypeError):
        raise MuseError("Muse config context must be an integer") from None
    bundle = inspect_model(model, context)
    out = Path(cfg.get("tokenizer") or "")
    expected = {"vocab.json", "merges.txt", "token_type.json", "tokenizer.json", "chat_template.jinja"}
    if not cfg.get("tokenizer") or any(not (out / name).is_file() for name in expected):
        raise MuseError("Muse tokenizer sidecars or chat template are missing; run strata launch --family muse --model-path again")
    tokens, merges, types, special, template = _tokenizer(bundle)
    try:
        tcfg = json.loads((out / "tokenizer.json").read_text(encoding="utf-8"))
        if (json.loads((out / "vocab.json").read_text(encoding="utf-8")) != {t: i for i, t in enumerate(tokens)} or
                json.loads((out / "token_type.json").read_text(encoding="utf-8")) != types or
                (out / "merges.txt").read_text(encoding="utf-8").splitlines() != merges or
                (out / "chat_template.jinja").read_text(encoding="utf-8") != template or
                tcfg.get("pre") != bundle.metadata["tokenizer.ggml.pre"] or tcfg.get("special_ids") != special):
            raise MuseError("Muse tokenizer sidecars do not match this GGUF; prepare them again with strata launch")
    except (OSError, ValueError) as exc:
        raise MuseError(str(exc)) from None
    return bundle
