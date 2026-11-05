
import math
import time
import numpy as np
import torch
import torch.nn.functional as F

from model.zero_model import ZEROTransformer


CHECKPOINT = "checkpoints/nepali/zero_ne_exp001_best.pt"
VALIDATION = "data/processed/wikipedia/ne/tokenized/validation.bin"

VOCAB_SIZE = 16000
SEQUENCE_LENGTH = 256
BATCH_SIZE = 4
PAD_ID = 0


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 55)
    print("PROJECT ZERO | NEPALI FULL VALIDATION")
    print("=" * 55)
    print("Device:", device)

    started = time.time()

    tokens = np.memmap(
        VALIDATION,
        dtype=np.uint16,
        mode="r"
    )

    if len(tokens) < 2:
        raise ValueError("Validation dataset is too small")

    if int(tokens.max()) >= VOCAB_SIZE:
        raise ValueError("Validation contains invalid token IDs")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    if checkpoint.get("config", {}).get("vocab_size") != VOCAB_SIZE:
        raise ValueError("Checkpoint vocabulary mismatch")

    model = ZEROTransformer(vocab_size=VOCAB_SIZE)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.to(device)
    model.eval()

    total_loss = 0.0
    total_targets = 0
    total_windows = 0

    starts = list(range(0, len(tokens) - 1, SEQUENCE_LENGTH))

    with torch.inference_mode():
        for batch_start in range(0, len(starts), BATCH_SIZE):
            batch_starts = starts[
                batch_start:batch_start + BATCH_SIZE
            ]

            x_batch = np.full(
                (len(batch_starts), SEQUENCE_LENGTH),
                PAD_ID,
                dtype=np.int64
            )

            y_batch = np.full(
                (len(batch_starts), SEQUENCE_LENGTH),
                -100,
                dtype=np.int64
            )

            for i, start in enumerate(batch_starts):
                end = min(
                    start + SEQUENCE_LENGTH,
                    len(tokens) - 1
                )

                x = np.asarray(
                    tokens[start:end],
                    dtype=np.int64
                )

                y = np.asarray(
                    tokens[start + 1:end + 1],
                    dtype=np.int64
                )

                x_batch[i, :len(x)] = x
                y_batch[i, :len(y)] = y

            input_ids = torch.from_numpy(x_batch).to(device)
            targets = torch.from_numpy(y_batch).to(device)

            logits = model(input_ids)

            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="sum"
            )

            total_loss += loss.item()
            total_targets += (targets != -100).sum().item()
            total_windows += len(batch_starts)

            if total_windows % 100 < BATCH_SIZE:
                print(
                    f"Evaluated windows: {total_windows}/{len(starts)}"
                )

    average_loss = total_loss / total_targets
    perplexity = math.exp(average_loss)

    duration = time.time() - started

    print("\n" + "=" * 55)
    print("ZERO NEPALI EXP-001 | FULL EVALUATION RESULTS")
    print("=" * 55)
    print("Checkpoint step:", checkpoint["step"])
    print("Parameters:", f'{checkpoint["parameter_count"]:,}')
    print("Validation tokens:", len(tokens))
    print("Evaluated target tokens:", total_targets)
    print("Evaluation windows:", total_windows)
    print("Average validation loss:", f"{average_loss:.6f}")
    print("Perplexity:", f"{perplexity:.4f}")
    print("Duration:", f"{duration:.2f} seconds")
    print("=" * 55)


if __name__ == "__main__":
    main()
