import unittest

from tools.strata_tokenizer import BYTE_TO_UNICODE, Tokenizer


class MuseTokenizer(unittest.TestCase):
    def test_gpt4o_splits_case_digits_contractions_and_unicode(self):
        tok = Tokenizer(list(BYTE_TO_UNICODE.values()), [], pre="gpt-4o")
        self.assertEqual(tok._split("HELLOthere 12345 it's αβ!\n"),
                         ["HELLOthere", " ", "123", "45", " it's", " αβ", "!\n"])
        self.assertEqual(tok._split("ΓΕΙΑσου Ångström ١٢٣٤"), ["ΓΕΙΑσου", " Ångström", " ", "١٢٣", "٤"])

    def test_alias_round_trip_and_special_ids(self):
        tokens = list(BYTE_TO_UNICODE.values()) + ["<|eot|>"]
        types = [1] * 256 + [3]
        corpus = ["hello 123456", "ÉCOLE déjà e\u0301", "日本語🙂\r\n", "\x00\t spaces  ", "<|eot|>"]
        for pre in ("gpt-4o", "llama4"):
            tok = Tokenizer(tokens, [], types, pre, {"eot": 256})
            for text in corpus:
                self.assertEqual(tok.decode(tok.encode(text, parse_special=True)), text)
            self.assertEqual(tok.encode("<|eot|>", parse_special=True), [256])


if __name__ == "__main__":
    unittest.main()
