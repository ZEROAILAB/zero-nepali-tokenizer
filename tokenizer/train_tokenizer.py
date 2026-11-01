from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


print("====================================")
print("       ZERO TOKENIZER v0.1")
print("====================================")

# Where our English training text lives
dataset_file = "data/raw/tinystories.txt"

# Where we will save ZERO's tokenizer
output_file = "tokenizer/zero_tokenizer.json"


print("Creating tokenizer...")

# Create an empty BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

# Split text into basic pieces first
tokenizer.pre_tokenizer = Whitespace()

# Special tokens
special_tokens = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>"
]

# Train the tokenizer
trainer = BpeTrainer(
    vocab_size=16000,
    min_frequency=2,
    special_tokens=special_tokens
)

print("Training tokenizer on ZERO dataset...")

tokenizer.train(
    [dataset_file],
    trainer
)

# Save tokenizer
tokenizer.save(output_file)

print("------------------------------------")
print("ZERO TOKENIZER READY")
print("------------------------------------")

print("Vocabulary size:", tokenizer.get_vocab_size())
print("Saved to:", output_file)
