import os
import numpy as np
import torch
import torch.nn.functional as F

from model.zero_model import ZEROTransformer

# ==============================
# ZERO 100M NEPALI TRAINER
# ==============================

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
EVAL_STEPS = 5

GRAD_CLIP = 1.0

# ==============================
# DEVICE
# ==============================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("================================")
print("ZERO 100M NEPALI TRAINER")
print("================================")
print("Device:", device)

# ==============================
# DATASET
# ==============================

print("\nLoading memory-mapped datasets...")

train_tokens = np.memmap(
    TRAIN_PATH,
    dtype=np.uint16,
    mode="r"
)

val_tokens = np.memmap(
    VAL_PATH,
    dtype=np.uint16,
    mode="r"
)

print("Training tokens:", len(train_tokens))
print("Validation tokens:", len(val_tokens))

if len(train_tokens) <= SEQUENCE_LENGTH:
    raise ValueError("Training dataset is too small")

if len(val_tokens) <= SEQUENCE_LENGTH:
    raise ValueError("Validation dataset is too small")

# ==============================
# MODEL
# ==============================

print("\nCreating Nepali ZERO model...")

model = ZEROTransformer(vocab_size=VOCAB_SIZE)
model = model.to(device)

parameters = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("Parameters:", f"{parameters:,}")

# ==============================
# OPTIMIZER
# ==============================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)

# ==============================
# CHECKPOINTS
# ==============================

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

latest_checkpoint = os.path.join(
    CHECKPOINT_DIR,
    "zero_ne_latest.pt"
)

best_checkpoint = os.path.join(
    CHECKPOINT_DIR,
    "zero_ne_best.pt"
)

start_step = 1
best_val_loss = float("inf")

if os.path.exists(latest_checkpoint):
    print("\nResuming Nepali training...")

    checkpoint = torch.load(
        latest_checkpoint,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_step = checkpoint["step"] + 1
    best_val_loss = checkpoint.get(
        "best_val_loss",
        float("inf")
    )

    print("Resumed at step:", start_step)

# ==============================
# BATCH GENERATOR
# ==============================

def get_batch(tokens):
    max_start = len(tokens) - SEQUENCE_LENGTH - 1

    starts = np.random.randint(
        0,
        max_start + 1,
        size=BATCH_SIZE
    )

    x = np.stack([
        np.asarray(
            tokens[s:s + SEQUENCE_LENGTH],
            dtype=np.int64
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

# ==============================
# EVALUATION
# ==============================

@torch.no_grad()
def evaluate():
    model.eval()
    losses = []

    for _ in range(EVAL_STEPS):
        x, y = get_batch(val_tokens)

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            y.reshape(-1)
        )

        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)

# ==============================
# TRAINING
# ==============================

print("\nStarting Nepali training...")

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
        model.parameters(),
        GRAD_CLIP
    )

    optimizer.step()

    if step == start_step or step % 2 == 0:
        print(
            f"Step {step:4d} | "
            f"Train Loss: {loss.item():.4f}"
        )

    if step % EVAL_EVERY == 0:
        val_loss = evaluate()

        print(
            f"Validation Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "vocab_size": VOCAB_SIZE,
                    "sequence_length": SEQUENCE_LENGTH,
                },
                best_checkpoint
            )

            print("NEW BEST NEPALI CHECKPOINT")

    if step % SAVE_EVERY == 0:
        torch.save(
            {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "vocab_size": VOCAB_SIZE,
                "sequence_length": SEQUENCE_LENGTH,
            },
            latest_checkpoint
        )

        print("Latest checkpoint saved.")

print("\nZERO NEPALI TRAINING FINISHED")
print("Best validation loss:", best_val_loss)
