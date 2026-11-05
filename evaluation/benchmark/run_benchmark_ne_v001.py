import json
import math
import time
import random
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from model.zero_model import ZEROTransformer


ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "evaluation/benchmark/prompts_ne_v001.json"
CHECKPOINT = ROOT / "checkpoints/nepali_exp_003/zero_ne_exp003_best.pt"
TOKENIZER_PATH = ROOT / "tokenizer/zero_ne_tokenizer.json"
OUTPUT = ROOT / "evaluation/benchmark/results_ne_v001.jsonl"

TEMPERATURE = 0.8
TOP_K = 40
MAX_NEW_TOKENS = 80
CONTEXT_LIMIT = 256
BASE_SEED = 42
VOCAB_SIZE = 16000
EOS_ID = 3


def main():
    if not PROMPTS.exists():
        raise FileNotFoundError(PROMPTS)
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(TOKENIZER_PATH)

    prompts = json.loads(PROMPTS.read_text(encoding="utf-8"))

    if len(prompts) != 50 or len({p["id"] for p in prompts}) != 50:
        raise ValueError("Benchmark prompt integrity check failed")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print("PROJECT ZERO | NEPALI BENCHMARK v0.1")
    print("=" * 60)
    print("Device:", device)
    print("Prompts:", len(prompts))
    print("Checkpoint:", CHECKPOINT.name)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    load_started = time.time()
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    if checkpoint.get("experiment") != "ZERO_NEPALI_EXP_003":
        raise ValueError("Checkpoint is not EXP-003")

    if checkpoint.get("config", {}).get("vocab_size") != VOCAB_SIZE:
        raise ValueError("Checkpoint vocabulary mismatch")

    model = ZEROTransformer(vocab_size=VOCAB_SIZE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    model_load_seconds = time.time() - load_started

    print("Checkpoint step:", checkpoint["step"])
    print("Parameters:", f'{checkpoint["parameter_count"]:,}')
    print("Model load seconds:", f"{model_load_seconds:.2f}")
    print("Generation settings:", {
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "max_new_tokens": MAX_NEW_TOKENS,
        "context_limit": CONTEXT_LIMIT,
        "base_seed": BASE_SEED
    })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    total_generated = 0
    total_generation_seconds = 0.0

    with OUTPUT.open("w", encoding="utf-8") as result_file:
        for index, item in enumerate(prompts):
            seed = BASE_SEED + index
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            input_ids = tokenizer.encode(item["prompt"]).ids
            if not input_ids:
                raise ValueError(f"Empty tokenized prompt: {item['id']}")

            generated = torch.tensor(
                [input_ids],
                dtype=torch.long,
                device=device
            )

            new_tokens = []
            generation_started = time.perf_counter()

            with torch.inference_mode():
                for _ in range(MAX_NEW_TOKENS):
                    context = generated[:, -CONTEXT_LIMIT:]
                    logits = model(context)
                    next_logits = logits[:, -1, :] / TEMPERATURE

                    k = min(TOP_K, next_logits.shape[-1])
                    values, _ = torch.topk(next_logits, k)
                    threshold = values[:, -1].unsqueeze(-1)
                    next_logits[next_logits < threshold] = float("-inf")

                    probabilities = torch.softmax(next_logits, dim=-1)

                    if not torch.isfinite(probabilities).all():
                        raise RuntimeError(
                            f"Non-finite probabilities for {item['id']}"
                        )

                    next_token = torch.multinomial(probabilities, 1)
                    token_id = next_token.item()

                    if token_id == EOS_ID:
                        break

                    new_tokens.append(token_id)
                    generated = torch.cat(
                        (generated, next_token),
                        dim=1
                    )

            latency = time.perf_counter() - generation_started
            completion = tokenizer.decode(new_tokens)

            generated_count = len(new_tokens)
            tokens_per_second = (
                generated_count / latency if latency > 0 else 0.0
            )

            result = {
                "id": item["id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "completion": completion,
                "prompt_token_count": len(input_ids),
                "generated_token_count": generated_count,
                "latency_seconds": round(latency, 4),
                "tokens_per_second": round(tokens_per_second, 4),
                "seed": seed,
                "temperature": TEMPERATURE,
                "top_k": TOP_K,
                "max_new_tokens": MAX_NEW_TOKENS,
                "checkpoint": checkpoint["experiment"],
                "checkpoint_step": checkpoint["step"],
                "device": str(device)
            }

            result_file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            result_file.flush()

            total_generated += generated_count
            total_generation_seconds += latency

            print(
                f"[{index + 1:02d}/50] {item['id']} "
                f"| {generated_count} tokens "
                f"| {latency:.2f}s"
            )

    average_latency = total_generation_seconds / len(prompts)
    overall_speed = (
        total_generated / total_generation_seconds
        if total_generation_seconds > 0 else 0
    )

    print("\n" + "=" * 60)
    print("BENCHMARK GENERATION COMPLETED")
    print("=" * 60)
    print("Prompts processed:", len(prompts))
    print("Generated tokens:", total_generated)
    print("Average latency:", f"{average_latency:.3f}s")
    print("Aggregate generation speed:", f"{overall_speed:.2f} tokens/sec")
    print("Total generation time:", f"{total_generation_seconds:.2f}s")
    print("Results:", OUTPUT)
    print("=" * 60)


if __name__ == "__main__":
    main()
