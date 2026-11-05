import unittest
from pathlib import Path
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "tokenizer" / "zero_ne_tokenizer.json"


class TestZeroNepaliTokenizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    def test_vocabulary_size(self):
        self.assertEqual(self.tokenizer.get_vocab_size(), 16000)

    def test_special_token_ids(self):
        expected = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }
        for token, token_id in expected.items():
            self.assertEqual(
                self.tokenizer.token_to_id(token), token_id
            )

    def test_nepali_round_trip(self):
        samples = [
            "नेपाल हाम्रो देश हो।",
            "कृत्रिम बुद्धिमत्ता र नेपाली भाषा",
            "शून्यबाट नयाँ प्रविधि निर्माण गरौँ।",
            "नमस्कार! तपाईंलाई कस्तो छ?",
        ]

        for sample in samples:
            with self.subTest(sample=sample):
                encoded = self.tokenizer.encode(sample)
                decoded = self.tokenizer.decode(encoded.ids)
                self.assertEqual(decoded, sample)


if __name__ == "__main__":
    unittest.main(verbosity=2)
