import os
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
