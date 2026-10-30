"""Exercise the normal Service with a real Muse process and synthetic weights.

Set STRATA_MUSE_TEST_EXE to an existing strata-muse build to run these checks.
No HTTP listener, GPU or model download is needed.
"""
import os
import tempfile
import threading
import unittest
from pathlib import Path

from serve.frontend import use_family
from serve.muse import MuseTemplate
from serve.server import Service, StrataEngine
from strata_control import muse
from tools.muse.chat_fixture import ANSWER, generate
from tools.strata_tokenizer import Tokenizer


@unittest.skipUnless(os.environ.get("STRATA_MUSE_TEST_EXE"), "set STRATA_MUSE_TEST_EXE to a built Muse executable")
class ResidentChat(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        model = generate(Path(self.tmp.name))
        bundle = muse.inspect_model(model, 1024)
        directory = muse.prepare_tokenizer(bundle, Path(self.tmp.name) / "tokenizer")
        tok = Tokenizer.from_gguf(model)
        args = ["--model", str(model), "--max-context", "1024", "--backend",
                os.environ.get("STRATA_MUSE_TEST_BACKEND", "cpu")]
        for env, flag in (("STRATA_MUSE_TEST_GPU", "--gpu"), ("STRATA_MUSE_TEST_CACHE_MIB", "--moe-cache-mib")):
            if os.environ.get(env):
                args += [flag, os.environ[env]]
        self.engine = StrataEngine(os.environ["STRATA_MUSE_TEST_EXE"], args)
        self.addCleanup(self.engine.close)
        self.addCleanup(use_family, None)
        self.svc = Service(self.engine, tok, MuseTemplate(directory / "chat_template.jinja", tok), model_name="synthetic-muse")
        self.svc.set_family("muse")

    def test_chat_replays_without_qwen_cache_or_markers(self):
        for text in ("hello", "second request αβ"):
            ids, thinking, cap = self.svc.prepare([{"role": "user", "content": text}], None, {}, max_new=1)
            output = list(self.svc.run(ids, thinking, None, cap, {"temperature": 0}, threading.Event()))
            visible = "".join(event.text for kind, event in output if kind == "event" and event.kind == "content")
            self.assertEqual(visible, ANSWER)
            done = next(data for kind, data in output if kind == "done")
            self.assertEqual(done["completion_tokens"], 1)
            self.assertEqual(self.engine.last["prompt_tokens"], len(ids))
            self.assertTrue(self.engine.alive())

    def test_tools_and_logprobs_fail_before_engine_generation(self):
        for tools, req in (([{"name": "exec"}], {}), (None, {"logprobs": True})):
            with self.assertRaises(ValueError):
                self.svc.prepare([{"role": "user", "content": "hello"}], tools, {}, max_new=1, req=req)
        self.assertEqual(self.engine.last, {})


if __name__ == "__main__":
    unittest.main()
