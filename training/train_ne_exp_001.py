import os
import json
import time
import random
import numpy as np
import torch
import torch.nn.functional as F

from tokenizers import Tokenizer
from model.zero_model import ZEROTransformer


# PROJECT ZERO | NEPALI EXP-001

TOKENIZER_PATH = "tokenizer/zero_ne_tokenizer.json"
TRAIN_PATH = "data/processed/wikipedia/ne/tokenized/train.bin"
VAL_PATH = "data/processed/wikipedia/ne/tokenized/validation.bin"

CHECKPOINT_DIR = "checkpoints/nepali"
VOCAB_SIZE = 16000

BATCH_SIZE = 1
SEQUENCE_LENGTH = 256
LEARNING_RATE = 1e-4
TRAINING_STEPS = 20
SAVE_EVERY = 10
EVAL_EVERY = 10
EVAL_STEPS = 20
GRAD_CLIP = 1.0
SEED = 42

CONFIG = {
    "experiment": "ZERO_NEPALI_EXP_001",
    "vocab_size": VOCAB_SIZE,
    "batch_size": BATCH_SIZE,
    "sequence_length": SEQUENCE_LENGTH,
    "learning_rate": LEARNING_RATE,
    "training_steps": TRAINING_STEPS,
    "eval_steps": EVAL_STEPS,
    "seed": SEED,
}

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LATEST = os.path.join(CHECKPOINT_DIR, "zero_ne_exp001_latest.pt")
BEST = os.path.join(CHECKPOINT_DIR, "zero_ne_exp001_best.pt")
HISTORY = os.path.join(CHECKPOINT_DIR, "zero_ne_exp001_history.json")

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
print("PROJECT ZERO | NEPALI EXP-001")
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


# Resume only from a compatible EXP-001 checkpoint.
if os.path.exists(LATEST):
    print("Loading previous EXP-001 checkpoint...")

    checkpoint = torch.load(
        LATEST,
        map_location="cpu",
        weights_only=False
    )

    if checkpoint.get("experiment") != CONFIG["experiment"]:
        raise ValueError("Checkpoint experiment mismatch")

    if checkpoint.get("config") != CONFIG:
        raise ValueError("Checkpoint configuration mismatch")

    if checkpoint.get("parameter_count") != parameter_count:
        raise ValueError("Checkpoint architecture mismatch")

    model.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    np_rng.bit_generator.state = checkpoint["numpy_rng_state"]
    torch.set_rng_state(checkpoint["torch_rng_state"])

    history = checkpoint.get("history", [])
    best_val_loss = checkpoint["best_val_loss"]
    start_step = checkpoint["step"] + 1

    print("Resumed from step:", start_step - 1)


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

print("\nZERO NEPALI EXP-001 FINISHED")
print("Best validation loss:", best_val_loss)
print("Duration:", f"{time.time() - started:.2f} seconds")
print("Checkpoint directory:", CHECKPOINT_DIR)
