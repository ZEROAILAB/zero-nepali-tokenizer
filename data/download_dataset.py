from pathlib import Path
from datasets import load_dataset

print("====================================")
print("       ZERO DATA ENGINE v0.2")
print("====================================")

OUTPUT_FILE = Path("data/raw/tinystories.txt")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("Downloading TinyStories...")

dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train[:10000]"
)

print("Saving dataset...")

with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    for item in dataset:
        text = item["text"].strip()
        if text:
            f.write(text + "\n\n")

print("------------------------------------")
print("ZERO DATASET READY")
print("------------------------------------")
print("Stories:", len(dataset))
print("Saved to:", OUTPUT_FILE)
