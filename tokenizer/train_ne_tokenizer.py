import json
from pathlib import Path
from tokenizers import Tokenizer, models, trainers
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFC

BASE = Path("data/processed/wikipedia/ne")
TRAIN = BASE / "train.jsonl"
TEXT = BASE / "train_tokenizer.txt"
OUTPUT = Path("tokenizer/zero_ne_tokenizer.json")

print("ZERO | NEPALI TOKENIZER TRAINING")
print("=" * 48)

count = 0

with TRAIN.open(encoding="utf-8") as src, TEXT.open("w", encoding="utf-8") as dst:
    for line in src:
        item = json.loads(line)
        title = item.get("title", "")
        text = item.get("text", "")
        if text.strip():
            dst.write(title + "\n" + text + "\n\n")
            count += 1

print("Training articles:", count)

tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
tokenizer.normalizer = NFC()
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

trainer = trainers.BpeTrainer(
    vocab_size=16000,
    min_frequency=2,
    special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
)

tokenizer.train([str(TEXT)], trainer)
tokenizer.save(str(OUTPUT))

print("=" * 48)
print("NEPALI TOKENIZER READY")
print("Vocabulary:", tokenizer.get_vocab_size())
print("Saved:", OUTPUT)

samples = [
    "नेपाल हाम्रो देश हो।",
    "कृत्रिम बुद्धिमत्ता र नेपाली भाषा",
    "शून्यबाट नयाँ प्रविधि निर्माण गरौँ।"
]

for sample in samples:
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded.ids)
    print("\nInput:", sample)
    print("Tokens:", encoded.tokens)
    print("Decoded:", decoded)

print("\nTOKENIZER TEST COMPLETE")
