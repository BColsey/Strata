"""Muse recipient framing is independent of Qwen's XML parser."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from serve.muse import MuseOutputParser, MuseTemplate, validate_request


class MuseChat(unittest.TestCase):
    def events(self, text, width):
        parser = MuseOutputParser(thinking=True)
        events = []
        for at in range(0, len(text), width):
            events.extend(parser.feed(text[at:at + width]))
        events.extend(parser.finish("stop"))
        return {kind: "".join(e.text for e in events if e.kind == kind) for kind in ("reasoning", "content")}

    def test_recipient_changes_at_every_chunk_boundary(self):
        text = (" to=self<|message|>private α<|eom|><|start|>assistant to=commentary<|message|>Working. "
                "<|eom|><|start|>assistant to=user<|message|>Done.<|eot|>ignored")
        for width in range(1, len(text) + 1):
            with self.subTest(width=width):
                self.assertEqual(self.events(text, width), {"reasoning": "private α", "content": "Working. Done."})

    def test_content_does_not_interpret_qwen_tags(self):
        self.assertEqual(self.events("<|message|><think>literal</think>", 1)["content"], "<think>literal</think>")

    def test_truncated_headers_never_become_content(self):
        for text in (" to=self", "<|start|>assistant to=", " to=user<|mes"):
            self.assertEqual(self.events(text, 1), {"reasoning": "", "content": ""})

    def test_unknown_recipient_and_large_header_are_errors(self):
        for text in (" to=exec<|message|>secret", "a" * 257, "a" * 257 + "<|message|>"):
            with self.assertRaises(ValueError):
                MuseOutputParser().feed(text)

    def test_template_uses_bos_and_native_reasoning_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chat_template.jinja"
            path.write_text("{{ bos_token }}{{ disable_reasoning }}{{ reasoning_effort }}", encoding="utf-8")
            template = MuseTemplate(path, SimpleNamespace(tokens=["<bos>"], special_ids={"bos": 0}))
            self.assertEqual(template.render([], enable_thinking=False, reasoning_effort="high"), "<bos>Truehigh")
            self.assertFalse(template.caps["supports_tools"])
            with self.assertRaises(ValueError):
                template.render([], reasoning_effort="unknown")

    def test_unsupported_capabilities_fail_explicitly(self):
        for messages, tools, request in (([], [{"name": "exec"}], {}),
                                          ([{"role": "tool", "content": "x"}], None, {}),
                                          ([{"content": [{"type": "image_url"}]}], None, {}),
                                          ([], None, {"logprobs": True}),
                                          ([], None, {"reasoning_budget_tokens": 100})):
            with self.assertRaises(ValueError):
                validate_request(messages, tools, request)


if __name__ == "__main__":
    unittest.main()
