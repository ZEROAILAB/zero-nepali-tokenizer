
import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
import torch
import torch.nn.functional as F

from tokenizers import Tokenizer
from model.zero_model import ZEROTransformer


# ============================================================
# ZERO GENERATION
# ============================================================

CHECKPOINT_PATH = "checkpoints/zero_step_100.pt"
TOKENIZER_PATH = "tokenizer/zero_tokenizer.json"

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 50


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
print("        ZERO 100M GENERATION")
print("====================================")

print("Device:", device)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)


# ============================================================
# CREATE MODEL
# ============================================================

print("Loading ZERO model...")

model = ZEROTransformer()


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


print(
    "Loaded training step:",
    checkpoint["step"]
)

print(
    "Training loss:",
    checkpoint["loss"]
)


# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate(prompt):

    # Convert prompt into token IDs
    encoded = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device
    )

    print("\nZERO:")
    print(prompt, end="")

    # Generate tokens one at a time
    for _ in range(MAX_NEW_TOKENS):

        # Keep only the latest context
        input_context = input_ids[:, -1024:]

        # Model prediction
        with torch.no_grad():

            logits = model(input_context)

        # Get prediction for final position
        logits = logits[:, -1, :]

        # Temperature
        logits = logits / TEMPERATURE

        # Top-K filtering
        values, indices = torch.topk(
            logits,
            TOP_K
        )

        filtered_logits = torch.full_like(
            logits,
            float("-inf")
        )

        filtered_logits.scatter_(
            1,
            indices,
            values
        )

        # Convert scores into probabilities
        probabilities = F.softmax(
            filtered_logits,
            dim=-1
        )

        # Sample next token
        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        # Add token to sequence
        input_ids = torch.cat(
            [
                input_ids,
                next_token
            ],
            dim=1
        )

        # Convert token back to text
        token_text = tokenizer.decode(
            [next_token.item()]
        )

        print(
            token_text,
            end="",
            flush=True
        )

    print("\n")


# ============================================================
# RUN ZERO
# ============================================================

prompt = "Once upon a time"

generate(prompt)

print("------------------------------------")
print("ZERO GENERATION COMPLETE")
print("------------------------------------")
