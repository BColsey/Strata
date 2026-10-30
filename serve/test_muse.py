"""Muse recipient framing is independent of Qwen's XML parser."""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from jinja2 import TemplateSyntaxError

from serve import frontend, responses
from serve.muse import MuseOutputParser, MuseTemplate, validate_request


# Literal syntax excerpts from the model-owned template, with a message-loop scaffold.
# Source: unsloth/Muse-Spark-Open-816B-A42B-GGUF@c00670568b6ce5e56a247f67091247984df2a01e,
# UD-IQ1_S first shard, tokenizer.chat_template (16,295 bytes).
# Full-template SHA256: 2341a2d3b1f1b0ac92b6237d46e2fff495109bd3305cacdea54726f5e98411db.
# Keep the published effort validation and generation block; no tokenizer/weight data.
PUBLISHED_TEMPLATE_EXCERPT = """
{%- set reasoning_effort = reasoning_effort | default('xhigh', true) -%}
{%- set reasoning_effort_map = {'low': 8, 'medium': 32, 'high': 256, 'xhigh': 512} -%}
{%- if not (reasoning_effort is string and reasoning_effort in reasoning_effort_map) -%}
    {{- raise_exception('Unsupported reasoning_effort: ' ~ reasoning_effort ~ '. Expected one of: low, medium, high, xhigh.') -}}
{%- endif -%}
{%- macro esc(value) -%}
    {{- value | string
        | replace('<|', '< |')
        | replace('<image>', '< image>')
        | replace('<video>', '< video>')
        | replace('<audio>', '< audio>') -}}
{%- endmacro -%}
{{- 'Reasoning strength: ' ~ reasoning_effort_map[reasoning_effort] ~ '.' -}}
{%- for message in messages -%}
    {%- generation %}
    {%- set reasoning = message.get('reasoning_content') or message.get('reasoning') -%}
    {%- if reasoning -%}
        {{- '<|start|>assistant to=self<|message|>' -}}{{- esc(reasoning) -}}{{- '<|eom|>' -}}
    {%- endif -%}
    {%- endgeneration %}
{%- endfor -%}
"""


class MuseChat(unittest.TestCase):
    def template(self, source=PUBLISHED_TEMPLATE_EXCERPT):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "chat_template.jinja"
        path.write_text(source, encoding="utf-8")
        self.template_path = path
        return MuseTemplate(path, SimpleNamespace(tokens=["<bos>"], special_ids={"bos": 0}))

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

    def test_private_followups_need_complete_message_framing(self):
        for suffix in ("<|message|>private followup", " to=user<|message|>private followup"):
            text = " to=self<|message|>private<|eom|>" + suffix
            for width in range(1, len(text) + 1):
                with self.subTest(width=width, suffix=suffix), self.assertRaises(ValueError):
                    self.events(text, width)
        with self.assertRaises(ValueError):
            self.events(" to=self<|message|>private<|start|>assistant to=user<|message|>x", 1)

    def test_published_generation_block_preserves_output(self):
        template = self.template()
        messages = [{"role": "assistant", "content": "answer", "reasoning_content": "α <|eot|>"}]
        self.assertEqual(template.render(messages),
                         "Reasoning strength: 512.<|start|>assistant to=self<|message|>α < |eot|><|eom|>")
        self.assertFalse(template.caps["supports_tools"])

    def test_generation_matches_hugging_face_scope_and_whitespace(self):
        template = self.template("A {% generation %}{% set value = 'β' %} {{ value }} {% endgeneration %}{{ value }} Z")
        # HF AssistantTracker uses a CallBlock: assignments stay inside the block.
        self.assertEqual(template.render([]), "A  β  Z")
        # The default Qwen environment still refuses this Muse-only syntax.
        with self.assertRaises(TemplateSyntaxError):
            frontend.ChatTemplate(self.template_path)

    def test_template_uses_bos_and_supported_efforts(self):
        template = self.template("{{ bos_token }}{{ reasoning_effort }}")
        self.assertEqual(template.render([], enable_thinking=True, reasoning_effort="xhigh"), "<bos>xhigh")
        for effort in ("unknown", "minimal"):
            with self.subTest(effort=effort), self.assertRaises(ValueError):
                template.render([], reasoning_effort=effort)
        for kwargs in ({"enable_thinking": False}, {"disable_reasoning": True}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "disabl.*reasoning"):
                template.render([], **kwargs)

    def test_muse_date_helper_does_not_change_the_default_environment(self):
        template = self.template("{{ strftime_now('%Y-%m-%d') }}")
        self.assertEqual(template.render([]), datetime.now().strftime("%Y-%m-%d"))
        self.assertNotIn("strftime_now", frontend.ChatTemplate(self.template_path).template.environment.globals)

    def test_api_efforts_keep_the_published_four_strengths(self):
        frontend.use_family("muse")
        self.addCleanup(frontend.use_family, None)
        template = self.template()
        for effort, strength in (("low", 8), ("medium", 32), ("high", 256), ("xhigh", 512),
                                 ("max", 512), ("maximum", 512)):
            with self.subTest(effort=effort):
                requests = (frontend.openai_to_messages({"messages": [], "reasoning_effort": effort})[2],
                            frontend.anthropic_to_messages({"messages": [], "output_config": {"effort": effort}})[2],
                            responses.template_kwargs({"reasoning": {"effort": effort}}, {}))
                for kwargs in requests:
                    self.assertEqual(template.render([], **kwargs), f"Reasoning strength: {strength}.")

    def test_api_requests_cannot_silently_disable_reasoning_or_use_minimal(self):
        frontend.use_family("muse")
        self.addCleanup(frontend.use_family, None)
        template = self.template()
        for effort in ("minimal", "none", "off", False):
            with self.subTest(effort=effort), self.assertRaises(ValueError):
                kwargs = frontend.openai_to_messages({"messages": [], "reasoning_effort": effort})[2]
                template.render([], **kwargs)
        for req in ({"chat_template_kwargs": {"enable_thinking": False}},
                    {"thinking": {"type": "disabled"}}):
            normalize = frontend.anthropic_to_messages if "thinking" in req else frontend.openai_to_messages
            with self.subTest(request=req), self.assertRaisesRegex(ValueError, "disabl.*reasoning"):
                template.render([], **normalize({"messages": [], **req})[2])

    def test_unsupported_capabilities_fail_explicitly(self):
        for messages, tools, request in (([], [{"name": "exec"}], {}),
                                          ([{"role": "tool", "content": "x"}], None, {}),
                                          ([{"content": [{"type": "image_url"}]}], None, {}),
                                          ([], None, {"logprobs": True}),
                                          ([], None, {"logprobs": 0}),
                                          ([], None, {"strata_checkpoint": False}),
                                          ([], None, {"experimental_speed_projection": False}),
                                          ([], None, {"messages": [{"role": "user", "content": [{"type": "input_audio"}]}]}),
                                          ([], None, {"input": [{"type": "input_file"}]}),
                                          ([], None, {"reasoning_budget_tokens": 100})):
            with self.assertRaises(ValueError):
                validate_request(messages, tools, request)


if __name__ == "__main__":
    unittest.main()
