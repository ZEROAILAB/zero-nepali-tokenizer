
import argparse
import torch
from tokenizers import Tokenizer
from model.zero_model import ZEROTransformer


CHECKPOINT = "checkpoints/nepali_exp_002/zero_ne_exp002_best.pt"
TOKENIZER = "tokenizer/zero_ne_tokenizer.json"


def main():
    parser = argparse.ArgumentParser(description="ZERO Nepali Text Generation")
    parser.add_argument("--prompt", type=str, default="नेपाल हाम्रो सुन्दर देश हो।")
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.tokens < 1 or args.temperature <= 0 or args.top_k < 1:
        raise ValueError("Tokens, temperature and top-k must be positive.")

    torch.manual_seed(args.seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("PROJECT ZERO | NEPALI INFERENCE")
    print("Device:", device)

    tokenizer = Tokenizer.from_file(TOKENIZER)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    model = ZEROTransformer(vocab_size=tokenizer.get_vocab_size())
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    encoded = tokenizer.encode(args.prompt)
    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device
    )

    print("\nPROMPT:", args.prompt)

    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(args.tokens):
            context = generated[:, -256:]
            logits = model(context)
            next_token_logits = logits[:, -1, :] / args.temperature

            k = min(args.top_k, next_token_logits.size(-1))
            values, _ = torch.topk(next_token_logits, k)
            threshold = values[:, -1].unsqueeze(-1)

            next_token_logits[next_token_logits < threshold] = float("-inf")

            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

            if next_token.item() == 3:
                break

            generated = torch.cat((generated, next_token), dim=1)

    output = tokenizer.decode(generated[0].tolist())

    print("\nZERO OUTPUT:")
    print(output)
    print("\nGENERATION COMPLETE")


if __name__ == "__main__":
    main()
