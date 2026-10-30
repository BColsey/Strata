"""Subprocess checks of the resident Muse protocol against the independent oracle.

Run with --binary /path/to/strata-muse. NumPy is the existing fixture dependency;
CTest reports a skip when it is absent. No model download or GPU is needed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    import numpy as np
except ImportError:
    print("Muse protocol tests need the existing fixture dependency NumPy")
    sys.exit(77)

from tools.muse.fixture import TOKENS, metadata, weights, write_model
from tools.muse.reference import forward, load_f32

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
parser.add_argument("--backend")
parser.add_argument("--gpu")
parser.add_argument("--moe-cache-mib")
args, remaining = parser.parse_known_args()
BINARY = str(Path(args.binary).resolve())
BACKEND_ARGS = [part for flag, value in (("--backend", args.backend), ("--gpu", args.gpu),
                                       ("--moe-cache-mib", args.moe_cache_mib))
                if value is not None for part in (flag, value)]


class Peer:
    def __init__(self, model):
        self.stderr = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(
            [BINARY, "--model", str(model), "--max-context", "2048", "--serve", *BACKEND_ARGS],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
            text=True, encoding="utf-8", bufsize=1,
        )
        self.lines = queue.Queue()

        def pump():
            for line in self.proc.stdout:
                self.lines.put(line.rstrip("\r\n"))
            self.lines.put(None)

        self.reader = threading.Thread(target=pump, daemon=True)
        self.reader.start()
        self.info = self.read()
        self.ready = self.read()
        if not self.info.startswith("INFO arch=muse-spark ") or self.ready != "READY 2048 stop":
            self.close()
            raise AssertionError((self.info, self.ready))

    def read(self, timeout=20):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise AssertionError("engine protocol response timed out") from None
        if line is None:
            self.stderr.seek(0)
            raise AssertionError("engine closed stdout: " + self.stderr.read().decode("utf-8", "replace"))
        return line

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def collect(self):
        lines = []
        while True:
            line = self.read()
            lines.append(line)
            if line.startswith(("DONE ", "ERR ")):
                return lines

    def request(self, line):
        self.send(line)
        return self.collect()

    def close(self):
        try:
            if self.proc.poll() is None and not self.proc.stdin.closed:
                self.send("QUIT")
            self.proc.wait(timeout=10)
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(timeout=5)
            self.proc.stdin.close()
            self.reader.join(timeout=2)
            self.proc.stdout.close()
            self.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Protocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="muse-protocol-")
        cls.directory = Path(cls.temp.name)
        cls.model = cls.directory / "model.gguf"
        meta = metadata()
        meta["muse-spark.context_length"] = 2048
        write_model(cls.model, meta, weights())
        cls.meta, cls.tensors = load_f32(cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def completed(self, lines, prompt, finish="length"):
        self.assertTrue(lines[-1].startswith("DONE "), lines)
        fields = lines[-1].split()
        self.assertEqual(len(fields), 6, lines)
        tokens = [int(line[2:]) for line in lines if line.startswith("T ")]
        self.assertEqual(int(fields[1]), len(tokens))
        self.assertEqual(int(fields[2]), len(prompt))
        self.assertTrue(all(math.isfinite(float(f)) and float(f) >= 0 for f in fields[3:5]))
        self.assertEqual(fields[5], finish)
        self.assertTrue(all(0 <= t < 64 for t in tokens))
        for line in lines[:-1]:
            self.assertTrue(line.startswith(("T ", "PP ")), line)
            if line.startswith("PP "):
                progress = line.split()
                self.assertEqual(len(progress), 3)
                self.assertLessEqual(int(progress[1]), len(prompt))
                self.assertEqual(int(progress[2]), len(prompt))
        return tokens

    def oracle(self, prompt, count, *, repeat=1, frequency=0, presence=0, last_n=64):
        history = list(prompt)
        result = []
        for _ in range(count):
            logits = forward(self.meta, self.tensors, history)[-1].astype(np.float64)
            counts = np.bincount(history[-last_n:] if last_n else [], minlength=64)
            for i, n in enumerate(counts):
                if n:
                    logits[i] = logits[i] * repeat if logits[i] <= 0 else logits[i] / repeat
                    logits[i] -= n * frequency + presence
            token = int(np.argmax(logits))
            result.append(token)
            history.append(token)
        return result

    @staticmethod
    def command(count, prompt=TOKENS, keys=""):
        return f"GEN {count}{keys} " + ",".join(map(str, prompt))

    def test_greedy_oracle_reference_cli_and_request_reset(self):
        expected = self.oracle(TOKENS, 4)
        output = self.directory / "reference-cli.json"
        subprocess.run([BINARY, "--model", str(self.model), "--tokens", ",".join(map(str, TOKENS)),
                        "--ctx", "32", "--steps", "4", "--logits-file", str(output), *BACKEND_ARGS],
                       check=True, capture_output=True, text=True, timeout=20)
        self.assertEqual(json.loads(output.read_text())["generated_tokens"], expected)
        with Peer(self.model) as peer:
            self.assertIn("batch_slots=0", peer.info)
            for _ in range(2):
                got = self.completed(peer.request(self.command(4)), TOKENS)
                self.assertEqual(got, expected)
                self.completed(peer.request(self.command(3, [9, 9, 2])), [9, 9, 2])

    def test_zero_generation_reads_prompt(self):
        with Peer(self.model) as peer:
            lines = peer.request(self.command(0))
            self.assertEqual(self.completed(lines, TOKENS), [])
            self.assertIn("PP 7 7", lines)

    def test_sampling_penalties_and_history_window(self):
        with Peer(self.model) as peer:
            for last_n in (0, 2, 64):
                keys = (" temperature=0.8 top_k=1 top_p=0.4 min_p=0.2 seed=2769"
                        f" penalty_repeat=1.5 penalty_freq=0.75 penalty_present=0.5 penalty_last_n={last_n}")
                expected = self.oracle(TOKENS, 6, repeat=1.5, frequency=0.75, presence=0.5, last_n=last_n)
                self.assertEqual(self.completed(peer.request(self.command(6, keys=keys)), TOKENS), expected)

    def test_seed_reproducibility_and_sampling_filters(self):
        with Peer(self.model) as peer:
            command = self.command(12, keys=" temperature=1.3 top_k=8 top_p=0.8 min_p=0.05 seed=12345")
            first = self.completed(peer.request(command), TOKENS)
            self.completed(peer.request(self.command(4, [2, 3])), [2, 3])
            self.assertEqual(self.completed(peer.request(command), TOKENS), first)
            picks = set()
            best = self.oracle(TOKENS, 1)
            for seed in range(1, 9):
                varied = self.command(1, keys=f" temperature=100 top_k=2 seed={seed}")
                picks.update(self.completed(peer.request(varied), TOKENS))
                for restriction in ("top_p=0.001", "min_p=1"):
                    cmd = self.command(1, keys=f" temperature=100 {restriction} seed={seed}")
                    self.assertEqual(self.completed(peer.request(cmd), TOKENS), best)
            self.assertEqual(len(picks), 2, "temperature/seed path did not sample both candidates")

    def test_eos_and_eot_are_protocol_stop(self):
        best = self.oracle(TOKENS, 1)[0]
        for kind in ("eos", "eot"):
            model = self.directory / f"{kind}.gguf"
            meta = metadata()
            meta["muse-spark.context_length"] = 2048
            meta[f"tokenizer.ggml.{kind}_token_id"] = best
            write_model(model, meta, weights())
            with Peer(model) as peer:
                self.assertEqual(self.completed(peer.request(self.command(4)), TOKENS, "stop"), [best])

    def test_malformed_and_unsupported_commands_recover(self):
        invalid = ["", "STOP 1", "QUIT 1", "GEN", "GEN 1", "GEN -1 1", "GEN +1 1", "GEN 2048 1",
                   "GEN 1 64", "GEN 1 1,", "GEN 1 1,,2", "GEN 1 1 2", "GEN 1 18446744073709551616",
                   "GEN 1 temperature=nan 1", "GEN 1 temperature=inf 1", "GEN 1 temperature=-1 1",
                   "GEN 1 top_p=0 1", "GEN 1 top_p=1.1 1", "GEN 1 min_p=-0.1 1",
                   "GEN 1 penalty_repeat=0 1", "GEN 1 top_k=1 top_k=2 1", "GEN 1 seed=1x 1",
                   "GENI 1 image 1", "BGEN 0 1 1", "VRAM", "SAVE checkpoint"]
        for key in ("pcie_frac=0.5", "logprobs=1", "allowed=1,2", "bias=1:2", "prefill=1", "ckpt=0",
                    "img_keys=ab", "pin=1", "cvec=0", "pipe=1", "aux_cpus=1", "spec_min_p=0.1"):
            invalid.append(f"GEN 1 {key} 1")
        with Peer(self.model) as peer:
            for command in invalid:
                response = peer.request(command)
                self.assertEqual(len(response), 1, (command, response))
                self.assertTrue(response[0].startswith("ERR "), (command, response))
            expected = self.oracle(TOKENS, 2)
            self.assertEqual(self.completed(peer.request(self.command(2)), TOKENS), expected)

    def test_oversized_line_is_discarded_and_recovers(self):
        with Peer(self.model) as peer:
            self.assertIn("16 MiB", peer.request("X" * (16 * 1024 * 1024 + 1))[0])
            self.completed(peer.request(self.command(1)), TOKENS)

    def test_error_line_cannot_inject_protocol_bytes(self):
        with Peer(self.model) as peer:
            response = peer.request("GEN 1 bad\r\0key=1 1")
            self.assertEqual(len(response), 1)
            self.assertTrue(response[0].startswith("ERR "))
            self.assertNotIn("\r", response[0])
            self.assertNotIn("\0", response[0])
            self.completed(peer.request(self.command(1)), TOKENS)

    def test_stop_during_prefill_and_next_request(self):
        prompt = [1] * 1024
        with Peer(self.model) as peer:
            peer.send(self.command(100, prompt))
            self.assertEqual(peer.read(), "PP 0 1024")
            peer.send("STOP")
            self.assertEqual(self.completed(peer.collect(), prompt, "cancel"), [])
            self.assertEqual(self.completed(peer.request(self.command(2)), TOKENS), self.oracle(TOKENS, 2))

    def test_stop_during_decode_and_no_queued_generation(self):
        with Peer(self.model) as peer:
            peer.send(self.command(1000, [1]))
            prefix = []
            while not prefix or not prefix[-1].startswith("T "):
                prefix.append(peer.read())
            # Busy admission happens before even parsing this malformed token
            # list, so a rejected request cannot allocate a second large prompt.
            peer.send("GEN 1 not-a-token")
            peer.send("STOP")
            lines = prefix
            rejected = False
            while not lines[-1].startswith("DONE "):
                line = peer.read()
                if line.startswith("ERR "):
                    self.assertIn("busy", line)
                    rejected = True
                else:
                    lines.append(line)
            self.assertTrue(rejected)
            self.assertLess(len(self.completed(lines, [1], "cancel")), 1000)
            self.completed(peer.request(self.command(1)), TOKENS)

    def test_quit_and_eof_cancel_active_request_and_exit(self):
        for eof in (False, True):
            with Peer(self.model) as peer:
                peer.send(self.command(100, [1] * 1024))
                self.assertEqual(peer.read(), "PP 0 1024")
                if eof:
                    peer.proc.stdin.close()
                else:
                    peer.send("QUIT")
                self.assertEqual(peer.proc.wait(timeout=10), 0)
                self.completed(peer.collect(), [1] * 1024, "cancel")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
