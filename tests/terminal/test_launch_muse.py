"""Muse local launch: actual GGUF headers, no downloads, GPU or server process."""
from __future__ import annotations

import json
import os
import struct
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch.catalog import Catalog
from strata_cli.launch.runner import LaunchError, Runner
from strata_cli.launch.settings import LaunchSettings
from strata_control import muse
from tests.terminal.launch_support import HW_12, Home, catalog, run_cli
from tools.gguf_writer import GGUFWriter
from tools.muse.fixture import metadata, weights


def write_model(path, changes=None, tensors=None):
    md = {**metadata(), "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "llama4",
          "tokenizer.ggml.tokens": [f"token{i}" for i in range(64)], "tokenizer.ggml.merges": [],
          "tokenizer.ggml.token_type": [1] * 62 + [3, 3], "tokenizer.ggml.bos_token_id": 62,
          "tokenizer.ggml.eos_token_id": 63, "tokenizer.chat_template": "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}"}
    md.update(changes or {})
    writer = GGUFWriter()
    for key, value in md.items():
        if value is None:
            continue
        if isinstance(value, list):
            kind = "array:" + ("string" if key.endswith(("tokens", "merges")) else "bool" if key.endswith("pattern") else "u32")
        else:
            kind = {str: "string", bool: "bool", int: "u32", float: "f32"}[type(value)]
        writer.add(key, value, kind)
    for name, value in (weights() if tensors is None else tensors).items():
        writer.add_f32(name, value)
        writer.tensors[-1].shape = list(reversed(value.shape))
    writer.write(path)
    return path


class MuseLaunch(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.cat = catalog(HW_12)
        self.model = write_model(self.home.base / "model.gguf")
        self.exe = self.home.base / "strata-muse"
        self.exe.write_text("local executable placeholder; never started", encoding="utf-8")
        self.exe.chmod(0o755)
        patch = mock.patch.dict(os.environ, {"STRATA_MUSE_EXE": str(self.exe), "STRATA_MUSE_BACKEND": "cpu"})
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.object(Catalog, "hardware", lambda s, fresh=False: dict(HW_12))
        patch.start()
        self.addCleanup(patch.stop)

    def runner(self, **changes):
        return Runner(LaunchSettings(family="muse", model="local", model_path=str(self.model), **changes), catalog=self.cat)

    def test_local_catalog_has_no_download_sizes_or_default_setup_menu_entry(self):
        setup = self.cat.strata.setup_module()
        self.assertEqual(setup.LOCAL_FILE_FAMILIES["muse"]["models"], {})
        self.assertTrue(setup.LOCAL_FILE_FAMILIES["muse"]["local_only"])
        self.assertNotIn("muse", setup.OWN_PATH_FAMILIES)
        self.assertNotIn("muse", setup.menu_families(False))

    def test_cli_dry_plan_uses_actual_geometry_and_does_not_prepare_or_download(self):
        before = set(self.home.base.rglob("*"))
        with mock.patch.object(Runner, "_spawn", side_effect=AssertionError("no processes in plan")):
            code, out, err = run_cli(["launch", "--family", "muse", "--model-path", str(self.model), "--context", "16", "--plan", "--output", "json"])
        self.assertEqual(code, 0, err + out)
        plan = json.loads(out)["plan"]
        self.assertEqual(plan["download"]["gb"], 0)
        self.assertEqual(plan["placement"]["kv_bytes"], (3 + 16) * 2 * 16 * 2 * 4)
        self.assertEqual(plan["placement"]["scratch_bytes"], muse.TILE_BYTES + (64 * 64 + 16) * 4)
        self.assertEqual(plan["engine"]["args"], ["--serve", "--model", str(self.model), "--max-context", "16", "--backend", "cpu"])
        self.assertNotIn("experts_in_vram", plan["placement"])
        self.assertEqual(set(self.home.base.rglob("*")), before)

    def test_shared_family_keeps_its_small_context_refusal(self):
        code, out, err = run_cli(["launch", "--model", "q2_0", "--context", "16", "--plan", "--output", "json"])
        self.assertNotEqual(code, 0)
        self.assertIn("too small", out + err)

    def test_config_and_sidecars_reach_existing_profile_route_without_acquisition(self):
        runner = self.runner(context=16)
        with mock.patch.object(runner, "_spawn", side_effect=AssertionError("local preparation does not spawn")):
            result = runner.run(start=False)
        cfg = json.loads(Path(result.config).read_text())
        self.assertEqual(cfg["exe"], str(self.exe))
        self.assertEqual(cfg["arch"], "muse")
        self.assertEqual(cfg["parallel"], 1)
        self.assertEqual(cfg["args"], ["--model", str(self.model), "--max-context", "16", "--backend", "cpu"])
        self.assertEqual(muse.validate_config(cfg).context, 16)
        tcfg = json.loads((Path(cfg["tokenizer"]) / "tokenizer.json").read_text())
        self.assertEqual(tcfg["special_ids"], {"bos": 62, "eos": 63})
        self.assertEqual(tcfg["pre"], "llama4")

    def test_cuda_keeps_physical_card_in_engine_args_without_qwen_remapping(self):
        with mock.patch.dict(os.environ, {"STRATA_MUSE_BACKEND": "cuda"}):
            runner = self.runner(context=16, gpus=[2])
            cfg = json.loads(runner.build_config(runner.ensure_model()).read_text())
        self.assertEqual(cfg["args"][-2:], ["--gpu", "2"])
        self.assertNotIn("gpu", cfg)
        self.assertNotIn("layer_split", cfg)
        self.assertEqual(cfg["backend"], "cuda")

    def test_local_package_entrypoint_is_found_without_environment_override(self):
        folder = self.cat.root / "engine-muse" / "0.1-linux-x86_64-cpu"
        folder.mkdir(parents=True)
        exe = folder / "strata-muse"
        exe.write_bytes(b"local build")
        exe.chmod(0o755)
        with mock.patch.dict(os.environ, {"STRATA_MUSE_EXE": ""}):
            self.assertEqual(self.runner().plan()["engine"]["exe"], str(exe))

    def test_unsupported_launch_choices_are_refused_before_any_process(self):
        for changes in ({"vision": "on"}, {"parallel": 2}, {"gpus": [0, 1]}, {"gpu_roles": {"0": "main"}},
                        {"kv": "q4_0"}, {"tune": "quick"}, {"drafting": "mtp"}, {"speed": "faster"}, {"extra": {"--batch": "2"}}):
            with self.subTest(changes=changes), self.assertRaises(LaunchError):
                self.runner(**changes).plan()
        with mock.patch.dict(os.environ, {"STRATA_MUSE_BACKEND": "hip"}), self.assertRaisesRegex(LaunchError, "CPU or CUDA"):
            self.runner().plan()

    def test_missing_tokenizer_or_template_is_refused_without_fallback(self):
        for key in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.token_type", "tokenizer.chat_template"):
            with self.subTest(key=key):
                write_model(self.model, {key: None})
                with self.assertRaises(muse.MuseError):
                    muse.inspect_model(self.model)

    def test_context_and_rope_are_checked_from_metadata(self):
        with self.assertRaisesRegex(muse.MuseError, "trained context"):
            muse.inspect_model(self.model, 33)
        write_model(self.model, {muse.PREFIX + "rope.scaling.type": "yarn"})
        with self.assertRaisesRegex(muse.MuseError, "Scaled RoPE"):
            muse.inspect_model(self.model, 16)

    def test_host_budget_is_enforced_without_reading_tensor_payloads(self):
        write_model(self.model, {muse.PREFIX + "context_length": 20000000})
        with self.assertRaisesRegex(muse.MuseError, "1 GiB"):
            muse.inspect_model(self.model, 20000000)
        write_model(self.model, {muse.PREFIX + "context_length": 60000000,
                                 muse.PREFIX + "attention.sliding_window_pattern": [True, True]})
        with self.assertRaisesRegex(muse.MuseError, "resident budget"):
            muse.inspect_model(self.model, 60000000)

    def test_custom_quant_encoding_is_refused_by_its_actual_tensor_directory(self):
        data = bytearray(self.model.read_bytes())
        name = b"token_embd.weight"
        at = data.index(name) + len(name) + 4 + 2 * 8
        struct.pack_into("<I", data, at, 42)  # Strata Q2_0, not a pinned GGML converter.
        self.model.write_bytes(data)
        with self.assertRaisesRegex(muse.MuseError, "encoding Q2_0"):
            muse.inspect_model(self.model)

    def test_duplicate_metadata_is_refused_before_payload_processing(self):
        data = bytearray(self.model.read_bytes())
        key_len = struct.unpack_from("<Q", data, 24)[0]
        value_at = 24 + 8 + key_len + 4
        value_len = struct.unpack_from("<Q", data, value_at)[0]
        first = data[24:value_at + 8 + value_len]
        count = struct.unpack_from("<Q", data, 16)[0]
        struct.pack_into("<Q", data, 16, count + 1)
        self.model.write_bytes(data[:24] + first + data[24:])
        with self.assertRaisesRegex(muse.MuseError, "Duplicate Muse metadata"):
            muse.inspect_model(self.model)

    def test_complete_split_is_validated_and_missing_shard_is_named(self):
        tensors = list(weights().items())
        paths = [self.home.base / f"split-{i:05d}-of-00002.gguf" for i in (1, 2)]
        for i, path in enumerate(paths):
            half = dict(tensors[:len(tensors)//2] if i == 0 else tensors[len(tensors)//2:])
            write_model(path, {"split.count": 2, "split.no": i, "split.tensors.count": len(tensors)}, half)
        self.assertEqual(muse.inspect_model(paths[1], 16).paths, tuple(paths))
        paths[1].unlink()
        with self.assertRaisesRegex(muse.MuseError, "split-00002-of-00002"):
            muse.inspect_model(paths[0])

    def test_split_identity_and_truncated_payload_are_refused(self):
        raw = self.model.read_bytes()
        self.model.write_bytes(raw[:-4])
        with self.assertRaisesRegex(muse.MuseError, "payload"):
            muse.inspect_model(self.model)
        write_model(self.model, {"split.count": 2})
        with self.assertRaisesRegex(muse.MuseError, "every shard"):
            muse.inspect_model(self.model)

    def test_stale_sidecars_and_injected_unsupported_config_flags_are_refused(self):
        runner = self.runner(context=16)
        cfg = json.loads(runner.ensure_model().read_text())
        for change in ({"gpu": 0}, {"effort_position": "end"}, {"reasoning_close_retry": True},
                       {"user_args": ["--mtp"]}, {"split_serve": {"prefill_gpu": 0, "decode_gpu": 1}}):
            with self.subTest(change=change), self.assertRaises(muse.MuseError):
                muse.validate_config({**cfg, **change})
        (Path(cfg["tokenizer"]) / "chat_template.jinja").write_text("wrong template")
        with self.assertRaisesRegex(muse.MuseError, "do not match"):
            muse.validate_config(cfg)
        for flags in (["--batch", "2"], ["--max-context", "20"], ["--vision", "on"]):
            with self.subTest(flags=flags), self.assertRaises(muse.MuseError):
                muse.validate_config({**cfg, "args": [*cfg["args"], *flags]})


if __name__ == "__main__":
    unittest.main()
