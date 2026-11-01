import torch
from tokenizers import Tokenizer
from model.zero_model import ZEROTransformer

TOKENIZER_PATH = "tokenizer/zero_tokenizer.json"
CHECKPOINT_PATH = "checkpoints/zero_step_50.pt"

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 40


# =========================
# DEVICE
# =========================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("================================")
print("        ZERO AI v0.1")
print("================================")
print("Device:", device)


# =========================
# TOKENIZER
# =========================

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)


# =========================
# MODEL
# =========================

print("Loading ZERO model...")

model = ZEROTransformer()

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print("Checkpoint step:", checkpoint.get("step"))
print("Checkpoint loss:", checkpoint.get("loss"))
print("Model loaded.")
print()


# =========================
# GENERATION
# =========================

@torch.no_grad()
def generate(prompt):

    encoded = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device
    )

    for _ in range(MAX_NEW_TOKENS):

        # Keep within model context
        input_context = input_ids[:, -1024:]

        logits = model(input_context)

        # Last token prediction
        logits = logits[:, -1, :]

        # Temperature
        logits = logits / TEMPERATURE

        # Top-K sampling
        values, indices = torch.topk(
            logits,
            min(TOP_K, logits.size(-1))
        )

        probabilities = torch.softmax(values, dim=-1)

        next_token = indices[
            0,
            torch.multinomial(probabilities[0], 1)
        ]

        next_token = next_token.view(1, 1)

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1
        )

    output_ids = input_ids[0].tolist()

    return tokenizer.decode(output_ids)


# =========================
# CHAT
# =========================

while True:

    prompt = input("You: ")

    if prompt.lower() in ["exit", "quit"]:
        print("ZERO: Shutdown.")
        break

    if not prompt.strip():
        continue

    print("\nZERO:", generate(prompt))
    print()
