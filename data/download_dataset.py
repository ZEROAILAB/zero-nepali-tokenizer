from datasets import load_dataset

print("====================================")
print("       ZERO DATA ENGINE v0.1")
print("====================================")

print("Downloading TinyStories...")

dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train[:10000]"
)

output_file = "data/raw/tinystories.txt"

print("Saving dataset...")

with open(output_file, "w", encoding="utf-8") as f:

    for item in dataset:

        text = item["text"].strip()

        if text:
            f.write(text)
            f.write("\n\n")

print("------------------------------------")
print("ZERO DATASET READY")
print("------------------------------------")

print("Stories:", len(dataset))
print("Saved to:", output_file)
