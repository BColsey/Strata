"""Muse Spark text chat: model-owned template and recipient/channel framing.

The protocol is documented by llama.cpp's common/parsers/muse-spark.cpp.
Tool recipients use ATEM, which this reference adapter deliberately refuses.
No Qwen tags or tool-recovery heuristics are applied to Muse text.
"""
from __future__ import annotations

import re

from serve.frontend import ChatTemplate, Event, OutputParser

START = "<|start|>assistant"
MESSAGE = "<|message|>"
ENDS = ("<|eom|>", "<|eot|>", "<|start|>")


class MuseTemplate(ChatTemplate):
    def __init__(self, path, tokenizer):
        special = tokenizer.special_ids
        bos = special.get("bos")
        if not isinstance(bos, int) or not 0 <= bos < len(tokenizer.tokens):
            raise ValueError("Muse tokenizer needs its model's BOS token ID")
        self.bos = tokenizer.tokens[bos]
        super().__init__(path)
        # ATEM tool calls require their own adapter; never advertise Qwen XML.
        self.caps.update(supports_tools=False, supports_tool_calls=False,
                         supports_parallel_tool_calls=False)

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        validate_request(messages, tools)
        kwargs["bos_token"] = self.bos
        if "enable_thinking" in kwargs:
            kwargs["disable_reasoning"] = kwargs.pop("enable_thinking") is False
        effort = kwargs.get("reasoning_effort")
        if effort not in (None, "", "minimal", "low", "medium", "high"):
            raise ValueError("Muse reasoning effort must be minimal, low, medium or high")
        return super().render(messages, tools=None, add_generation_prompt=add_generation_prompt, **kwargs)


def validate_request(messages, tools=None, request=None):
    if tools or any(m.get("role") == "tool" or m.get("tool_calls") for m in messages):
        raise ValueError("Muse reference chat does not yet support ATEM tool calls")
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(p.get("type") not in (None, "text", "input_text", "output_text")
                                             for p in content if isinstance(p, dict)):
            raise ValueError("Muse reference chat supports text inputs only")
    req = request or {}
    unsupported = ("logprobs", "top_logprobs", "logit_bias", "strata_tune", "strata_prefix",
                   "reasoning_budget_tokens", "experimental_speed_projection")
    if any(req.get(k) not in (None, False, {}, []) for k in unsupported):
        raise ValueError("Muse reference chat does not support logprobs, logit bias, cache tuning or reasoning budgets")


class MuseOutputParser(OutputParser):
    """Incremental recipient framing; private `to=self` stays reasoning.

    Header bytes are bounded and never emitted as answer text, including a
    header cut short by cancellation or the token limit.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._header = True
        self._ended = False
        self.state = "content"

    @property
    def clean(self):
        return not self._header and not self.buf

    def feed(self, delta):
        if self._ended:
            return []
        self.buf += delta
        out = []
        while self.buf:
            if self._header:
                at = self.buf.find(MESSAGE)
                if at > 256:
                    raise ValueError("Muse output header exceeds 256 characters")
                if at < 0:
                    if len(self.buf) > 256:
                        raise ValueError("Muse output header exceeds 256 characters")
                    break
                header = self.buf[:at]
                if header.startswith(START):
                    header = header[len(START):]
                match = re.fullmatch(r"\s*(?:to=(self|user|commentary))?\s*", header)
                if match is None:
                    raise ValueError("Muse output has an unsupported recipient or malformed header")
                self.state = "reasoning" if match[1] == "self" else "content"
                self.buf = self.buf[at + len(MESSAGE):]
                self._header = False
                continue
            found = [(self.buf.find(end), end) for end in ENDS if end in self.buf]
            if found:
                at, end = min(found)
                if at:
                    out.append(Event(self.state, self.buf[:at]))
                self.buf = self.buf[at + len(end):]
                if end == "<|eot|>":
                    self.buf, self._ended = "", True
                    break
                # A new message must include the assistant role before its
                # recipient, whether preceded by EOM or directly by START.
                if end == "<|start|>":
                    self.buf = "<|start|>" + self.buf
                self._header = True
                continue
            hold = max((n for end in ENDS for n in range(1, len(end)) if self.buf.endswith(end[:n])), default=0)
            text = self.buf[:-hold] if hold else self.buf
            if text:
                out.append(Event(self.state, text))
            self.buf = self.buf[-hold:] if hold else ""
            break
        return out

    def finish(self, reason=None):
        out = []
        if not self._header and self.buf and not any(end.startswith(self.buf) for end in ENDS):
            out.append(Event(self.state, self.buf))
        self.buf = ""
        return out
