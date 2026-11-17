import unicodedata
import unittest

from tokenizers import Tokenizer


class TestZeroTokenizerV02(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(
            "tokenizer/zero_tokenizer_v02.json"
        )

    def test_round_trip(self):
        samples = [
            "Hello, world!",
            "Once upon a time, there was a little girl.",
            "ZERO AI: intelligence at ZERO.",
            "Line one.\n\nLine two!",
            "नेपाल सुन्दर छ। नमस्ते, संसार!",
            "Price: $100. Version 0.2.",
        ]

        for text in samples:
            with self.subTest(text=text):
                encoded = self.tokenizer.encode(text)
                decoded = self.tokenizer.decode(encoded.ids)
                self.assertEqual(
                    decoded,
                    unicodedata.normalize("NFC", text),
                )

    def test_special_token_ids(self):
        vocab = self.tokenizer.get_vocab()

        self.assertEqual(vocab["<PAD>"], 0)
        self.assertEqual(vocab["<UNK>"], 1)
        self.assertEqual(vocab["<BOS>"], 2)
        self.assertEqual(vocab["<EOS>"], 3)

    def test_vocabulary_size(self):
        self.assertEqual(
            self.tokenizer.get_vocab_size(),
            12710,
        )


if __name__ == "__main__":
    unittest.main()