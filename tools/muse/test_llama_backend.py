"""Optional compiled-adapter checks; no model download and no GPU required."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
from tools.muse.fixture import TOKENS, metadata, weights, write_model
from tools.muse.reference import forward, load_f32


@unittest.skipUnless(os.environ.get("STRATA_MUSE_LLAMA_TEST_EXE"), "set STRATA_MUSE_LLAMA_TEST_EXE to the built adapter")
class LlamaBackend(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "model.gguf"
        write_model(self.model, metadata(), weights())
        self.output = self.root / "logits.json"

    def run_engine(self, *extra, tokens=TOKENS, output=None):
        return subprocess.run([os.environ["STRATA_MUSE_LLAMA_TEST_EXE"], "--model", str(self.model),
                               "--tokens", ",".join(map(str, tokens)), "--ctx", "32", "--steps", "4",
                               "--logits-file", str(output or self.output), *extra],
                              capture_output=True, text=True, timeout=30)

    def test_full_logits_and_greedy_tokens_match_independent_oracle(self):
        result = self.run_engine("--threads", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = json.loads(self.output.read_text())
        md, tensors = load_f32(self.model)
        expected = forward(md, tensors, TOKENS)
        values = np.asarray(actual["logits"])
        self.assertEqual(actual["tokens"], TOKENS)
        self.assertEqual(values.shape, expected.shape)
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_allclose(values, expected, atol=2e-5, rtol=2e-5)
        history, generated = list(TOKENS), []
        for _ in range(4):
            token = int(np.argmax(forward(md, tensors, history)[-1]))
            history.append(token)
            generated.append(token)
        self.assertEqual(actual["generated_tokens"], generated)

    def test_output_cannot_overwrite_model_or_hardlink(self):
        alias = self.root / "alias.gguf"
        os.link(self.model, alias)
        digest = hashlib.sha256(self.model.read_bytes()).hexdigest()
        for output in (self.model, alias):
            result = self.run_engine(output=output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("aliases a model shard", result.stderr)
            self.assertEqual(hashlib.sha256(self.model.read_bytes()).hexdigest(), digest)

    def test_invalid_options_do_not_fall_back_to_reference_or_other_devices(self):
        for args in (("--backend", "cuda"), ("--threads", "0"), ("--threads", "1025"),
                     ("--moe-cache-mib", "1"), ("--gpu", "0"), ("--max-context", "32"),
                     ("--backend", "llama-cuda", "--gpu", "0,0"),
                     ("--backend", "llama-cuda", "--gpu", "2147483648")):
            with self.subTest(args=args):
                result = self.run_engine(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())
        result = self.run_engine(tokens=[64])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside vocabulary", result.stderr)

    def test_wrong_architecture_is_refused_before_model_execution(self):
        write_model(self.model, {**metadata(), "general.architecture": "qwen35moe"}, weights())
        result = self.run_engine()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("architecture must be muse-spark", result.stderr)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
