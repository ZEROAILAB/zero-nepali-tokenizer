import os
import json
import time
import random
import numpy as np
import torch
import torch.nn.functional as F

from tokenizers import Tokenizer
from model.zero_model import ZEROTransformer


# PROJECT ZERO | NEPALI EXP-003

TOKENIZER_PATH = "tokenizer/zero_ne_tokenizer.json"
TRAIN_PATH = "data/processed/wikipedia/ne/tokenized/train.bin"
VAL_PATH = "data/processed/wikipedia/ne/tokenized/validation.bin"

CHECKPOINT_DIR = "checkpoints/nepali_exp_003"
VOCAB_SIZE = 16000

BATCH_SIZE = 1
SEQUENCE_LENGTH = 256
LEARNING_RATE = 5e-5
TRAINING_STEPS = 5000
SAVE_EVERY = 250
EVAL_EVERY = 250
EVAL_STEPS = 20
GRAD_CLIP = 1.0
SEED = 43

CONFIG = {
    "experiment": "ZERO_NEPALI_EXP_003",
    "vocab_size": VOCAB_SIZE,
    "batch_size": BATCH_SIZE,
    "sequence_length": SEQUENCE_LENGTH,
    "learning_rate": LEARNING_RATE,
    "training_steps": TRAINING_STEPS,
    "eval_steps": EVAL_STEPS,
    "seed": SEED,
}

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LATEST = os.path.join(CHECKPOINT_DIR, "zero_ne_exp003_latest.pt")
BEST = os.path.join(CHECKPOINT_DIR, "zero_ne_exp003_best.pt")
HISTORY = os.path.join(CHECKPOINT_DIR, "zero_ne_exp003_history.json")

random.seed(SEED)
np_rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("=" * 52)
print("PROJECT ZERO | NEPALI EXP-003")
print("=" * 52)
print("Device:", device)
print("Seed:", SEED)

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
actual_vocab = tokenizer.get_vocab_size()

if actual_vocab != VOCAB_SIZE:
    raise ValueError(
        f"Tokenizer vocabulary mismatch: {actual_vocab} != {VOCAB_SIZE}"
    )

train_tokens = np.memmap(TRAIN_PATH, dtype=np.uint16, mode="r")
val_tokens = np.memmap(VAL_PATH, dtype=np.uint16, mode="r")

for name, tokens in [
    ("TRAIN", train_tokens),
    ("VALIDATION", val_tokens),
]:
    if len(tokens) <= SEQUENCE_LENGTH:
        raise ValueError(f"{name} dataset is too small")

    if int(tokens.max()) >= VOCAB_SIZE:
        raise ValueError(f"{name} contains out-of-range token IDs")

    print(f"{name} tokens:", len(tokens))

# Fixed validation positions for repeatable comparisons.
eval_rng = np.random.default_rng(SEED + 1)
max_val_start = len(val_tokens) - SEQUENCE_LENGTH - 1
eval_starts = eval_rng.integers(
    0, max_val_start + 1, size=EVAL_STEPS
)

model = ZEROTransformer(vocab_size=VOCAB_SIZE).to(device)

parameter_count = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

print("Parameters:", f"{parameter_count:,}")
print("Vocabulary:", VOCAB_SIZE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)

history = []
best_val_loss = float("inf")
start_step = 1


def atomic_save(payload, path):
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(step, train_loss):
    return {
        "experiment": CONFIG["experiment"],
        "config": CONFIG,
        "step": step,
        "train_loss": train_loss,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "parameter_count": parameter_count,
        "history": history,
        "numpy_rng_state": np_rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
    }


def get_batch(tokens):
    max_start = len(tokens) - SEQUENCE_LENGTH - 1

    starts = np_rng.integers(
        0, max_start + 1, size=BATCH_SIZE
    )

    x = np.stack([
        np.asarray(
            tokens[s:s + SEQUENCE_LENGTH], dtype=np.int64
        )
        for s in starts
    ])

    y = np.stack([
        np.asarray(
            tokens[s + 1:s + SEQUENCE_LENGTH + 1],
            dtype=np.int64
        )
        for s in starts
    ])

    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(y).to(device)
    )


@torch.inference_mode()
def evaluate():
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for start in eval_starts:
        x = torch.from_numpy(
            np.asarray(
                val_tokens[start:start + SEQUENCE_LENGTH],
                dtype=np.int64
            )
        ).unsqueeze(0).to(device)

        y = torch.from_numpy(
            np.asarray(
                val_tokens[start + 1:start + SEQUENCE_LENGTH + 1],
                dtype=np.int64
            )
        ).unsqueeze(0).to(device)

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            y.reshape(-1),
            reduction="sum"
        )

        total_loss += loss.item()
        total_tokens += y.numel()

    model.train()
    return total_loss / total_tokens


# Initialize EXP-003 from the best EXP-002 model weights.
# The optimizer is intentionally fresh.

SOURCE_CHECKPOINT = "checkpoints/nepali_exp_002/zero_ne_exp002_best.pt"

print("\\nLoading EXP-002 starting weights...")

source = torch.load(
    SOURCE_CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

if source.get("experiment") != "ZERO_NEPALI_EXP_002":
    raise ValueError("Expected a ZERO NEPALI EXP-002 checkpoint")

if source.get("config", {}).get("vocab_size") != VOCAB_SIZE:
    raise ValueError("Source vocabulary mismatch")

if source.get("parameter_count") != parameter_count:
    raise ValueError("Source architecture mismatch")

model.load_state_dict(
    source["model_state_dict"],
    strict=True
)

print("Loaded source checkpoint step:", source["step"])
print("Fresh AdamW optimizer initialized.")

# Measure EXP-003's starting point using EXP-002's fixed validation windows.
initial_val_loss = evaluate()
best_val_loss = initial_val_loss

history.append({
    "step": 0,
    "train_loss": None,
    "validation_loss": initial_val_loss,
    "source_experiment": "ZERO_NEPALI_EXP_002",
    "source_step": source["step"],
})

atomic_save(
    checkpoint_payload(0, None),
    BEST
)

with open(HISTORY, "w", encoding="utf-8") as file:
    json.dump(history, file, indent=2)

print(f"EXP-003 starting validation loss: {initial_val_loss:.6f}")
print("Initial EXP-003 checkpoint saved.")
start_step = 1

print("\nStarting Nepali training...")
started = time.time()
last_train_loss = None

model.train()

for step in range(start_step, TRAINING_STEPS + 1):
    x, y = get_batch(train_tokens)

    optimizer.zero_grad(set_to_none=True)

    logits = model(x)

    loss = F.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE),
        y.reshape(-1)
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(), GRAD_CLIP
    )

    optimizer.step()

    last_train_loss = loss.item()

    record = {
        "step": step,
        "train_loss": last_train_loss,
        "validation_loss": None,
    }

    print(
        f"Step {step:4d} | Train Loss: {last_train_loss:.4f}"
    )

    if step % EVAL_EVERY == 0:
        val_loss = evaluate()
        record["validation_loss"] = val_loss

        print(f"Validation Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            atomic_save(
                checkpoint_payload(step, last_train_loss),
                BEST
            )
            print("NEW BEST CHECKPOINT")

    history.append(record)

    if step % SAVE_EVERY == 0:
        atomic_save(
            checkpoint_payload(step, last_train_loss),
            LATEST
        )

        with open(HISTORY, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

        print("Latest checkpoint and history saved.")

print("\nZERO NEPALI EXP-002 FINISHED")
print("Best validation loss:", best_val_loss)
print("Duration:", f"{time.time() - started:.2f} seconds")
print("Checkpoint directory:", CHECKPOINT_DIR)
