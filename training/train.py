import os
import math
import torch
import torch.nn.functional as F

from model.zero_model import ZEROTransformer
from tokenizers import Tokenizer


# ============================================================
# ZERO 100M TRAINER V2
# ============================================================

TOKENIZER_PATH = "tokenizer/zero_tokenizer.json"

TRAIN_PATH = "data/processed/train.txt"
VAL_PATH = "data/processed/val.txt"

CHECKPOINT_DIR = "checkpoints"

BATCH_SIZE = 2
SEQUENCE_LENGTH = 128

LEARNING_RATE = 1e-4
TRAINING_STEPS = 1000

SAVE_EVERY = 100
EVAL_EVERY = 100
EVAL_STEPS = 20

GRAD_CLIP = 1.0


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


print("==============================================")
print("          ZERO 100M TRAINER V2")
print("==============================================")
print("Device:", device)


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

print("Tokenizer loaded.")


# ============================================================
# LOAD DATA
# ============================================================

def load_tokens(path):

    print(f"Loading: {path}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    encoded = tokenizer.encode(text)

    tokens = torch.tensor(
        encoded.ids,
        dtype=torch.long
    )

    print("Characters:", len(text))
    print("Tokens:", len(tokens))

    return tokens


train_tokens = load_tokens(TRAIN_PATH)
val_tokens = load_tokens(VAL_PATH)


# ============================================================
# MODEL
# ============================================================

print("\nCreating ZERO model...")

model = ZEROTransformer()
model = model.to(device)

parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Parameters:", parameters)
print("Parameters (M):", parameters / 1_000_000)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)


# ============================================================
# CHECKPOINT
# ============================================================

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

start_step = 1
best_val_loss = float("inf")

latest_checkpoint = os.path.join(
    CHECKPOINT_DIR,
    "zero_latest.pt"
)

best_checkpoint = os.path.join(
    CHECKPOINT_DIR,
    "zero_best.pt"
)


# ============================================================
# RESUME
# ============================================================

if os.path.exists(latest_checkpoint):

    print("\nLoading latest checkpoint...")

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

    model.to(device)

    print("Resumed from step:", start_step - 1)
    print("Best validation loss:", best_val_loss)

else:

    print("\nStarting from fresh model.")


# ============================================================
# BATCH
# ============================================================

def get_batch(tokens):

    max_start = len(tokens) - SEQUENCE_LENGTH - 1

    starts = torch.randint(
        0,
        max_start,
        (BATCH_SIZE,)
    )

    x = torch.stack([
        tokens[
            start:start + SEQUENCE_LENGTH
        ]
        for start in starts
    ])

    y = torch.stack([
        tokens[
            start + 1:start + SEQUENCE_LENGTH + 1
        ]
        for start in starts
    ])

    return x.to(device), y.to(device)


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate():

    model.eval()

    losses = []

    for _ in range(EVAL_STEPS):

        x, y = get_batch(val_tokens)

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                logits.size(-1)
            ),
            y.reshape(-1)
        )

        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


# ============================================================
# TRAINING
# ============================================================

model.train()

print("\n==============================================")
print("STARTING ZERO TRAINING")
print("==============================================")

for step in range(start_step, TRAINING_STEPS + 1):

    x, y = get_batch(train_tokens)

    optimizer.zero_grad(set_to_none=True)

    logits = model(x)

    loss = F.cross_entropy(
        logits.reshape(
            -1,
            logits.size(-1)
        ),
        y.reshape(-1)
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRAD_CLIP
    )

    optimizer.step()


    # ========================================================
    # TRAINING LOG
    # ========================================================

    if step == start_step or step % 10 == 0:

        print(
            f"Step {step:5d} | "
            f"Train Loss: {loss.item():.4f}"
        )


    # ========================================================
    # VALIDATION
    # ========================================================

    if step % EVAL_EVERY == 0:

        val_loss = evaluate()

        print(
            f"Step {step:5d} | "
            f"Validation Loss: {val_loss:.4f}"
        )


        # ====================================================
        # BEST MODEL
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": loss.item(),
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                },
                best_checkpoint
            )

            print(
                f"🔥 NEW BEST MODEL | "
                f"Val Loss: {val_loss:.4f}"
            )


    # ========================================================
    # LATEST CHECKPOINT
    # ========================================================

    if step % SAVE_EVERY == 0:

        torch.save(
            {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": loss.item(),
                "best_val_loss": best_val_loss,
            },
            latest_checkpoint
        )

        print(
            f"Checkpoint saved: {latest_checkpoint}"
        )


# ============================================================
# FINISHED
# ============================================================

print("\n==============================================")
print("ZERO TRAINING COMPLETE")
print("==============================================")

print("Best validation loss:", best_val_loss)import os
import torch
import torch.nn.functional as F

from model.zero_model import ZEROTransformer
from tokenizers import Tokenizer


# ============================================================
# ZERO 100M TRAINING CONFIG
# ============================================================

TOKENIZER_PATH = "tokenizer/zero_tokenizer.json"
DATA_PATH = "data/raw/tinystories.txt"

CHECKPOINT_DIR = "checkpoints"

BATCH_SIZE = 2
SEQUENCE_LENGTH = 128

LEARNING_RATE = 1e-4
TRAINING_STEPS = 500

SAVE_EVERY = 50


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


print("====================================")
print("       ZERO 100M TRAINING")
print("====================================")

print("Device:", device)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

with open(
    DATA_PATH,
    "r",
    encoding="utf-8"
) as f:

    text = f.read()


print("Characters:", len(text))


# ============================================================
# TOKENIZE DATA
# ============================================================

print("Tokenizing dataset...")

encoded = tokenizer.encode(text)

tokens = torch.tensor(
    encoded.ids,
    dtype=torch.long
)

print("Total tokens:", len(tokens))


# ============================================================
# MODEL
# ============================================================

print("\nCreating ZERO model...")

model = ZEROTransformer()

model = model.to(device)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

model.train()

print("\nStarting training...")
print("------------------------------------")


for step in range(1, TRAINING_STEPS + 1):

    # --------------------------------------------------------
    # Pick random starting positions
    # --------------------------------------------------------

    starts = torch.randint(
        0,
        len(tokens) - SEQUENCE_LENGTH - 1,
        (BATCH_SIZE,)
    )

    # --------------------------------------------------------
    # Create input and target batches
    # --------------------------------------------------------

    x = torch.stack(
        [
            tokens[start:start + SEQUENCE_LENGTH]
            for start in starts
        ]
    )

    y = torch.stack(
        [
            tokens[start + 1:start + SEQUENCE_LENGTH + 1]
            for start in starts
        ]
    )

    x = x.to(device)
    y = y.to(device)

    # --------------------------------------------------------
    # Clear old gradients
    # --------------------------------------------------------

    optimizer.zero_grad()

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    logits = model(x)

    # --------------------------------------------------------
    # Calculate loss
    # --------------------------------------------------------

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1)
    )

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    loss.backward()

    # --------------------------------------------------------
    # Update model
    # --------------------------------------------------------

    optimizer.step()

    # --------------------------------------------------------
    # Print progress
    # --------------------------------------------------------

    if step == 1 or step % 10 == 0:

        print(
            f"Step {step:4d} | "
            f"Loss: {loss.item():.4f}"
        )

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    if step % SAVE_EVERY == 0:

        os.makedirs(
            CHECKPOINT_DIR,
            exist_ok=True
        )

        checkpoint_path = (
            f"{CHECKPOINT_DIR}/"
            f"zero_step_{step}.pt"
        )

        torch.save(
            {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            },
            checkpoint_path
        )

        print(
            f"Checkpoint saved: "
            f"{checkpoint_path}"
        )


# ============================================================
# FINISHED
# ============================================================

print("------------------------------------")
print("ZERO TRAINING TEST COMPLETE")
print("------------------------------------")
