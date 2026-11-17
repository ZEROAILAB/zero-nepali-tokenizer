from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFC

print("====================================")
print("       ZERO TOKENIZER v0.2")
print("====================================")

DATASET = Path("data/raw/tinystories.txt")
OUTPUT = Path("tokenizer/zero_tokenizer_v02.json")

if not DATASET.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET}")

tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

tokenizer.normalizer = NFC()
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=16000,
    min_frequency=2,
    special_tokens=[
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>",
    ],
    initial_alphabet=ByteLevel.alphabet(),
)

print("Training on TinyStories...")
tokenizer.train([str(DATASET)], trainer)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
tokenizer.save(str(OUTPUT))

print("------------------------------------")
print("ZERO TOKENIZER v0.2 READY")
print("------------------------------------")
print("Vocabulary:", tokenizer.get_vocab_size())
print("Saved:", OUTPUT)