# ZERO Nepali Tokenizer

**Developed by ZEROAILAB**

An experimental Nepali-first tokenizer using Byte-Level Byte Pair Encoding (BPE).

## Version

- Release: v0.1
- Vocabulary: 16,000 tokens
- Normalization: Unicode NFC
- Special tokens: PAD, UNK, BOS, EOS

## Special Tokens

| Token | ID |
|---|---:|
| `<PAD>` | 0 |
| `<UNK>` | 1 |
| `<BOS>` | 2 |
| `<EOS>` | 3 |

## Installation

Python 3.9 or newer is recommended.

1. Clone this repository.
2. Enter the repository directory.
3. Create and activate a Python virtual environment.
4. Install dependencies using `python -m pip install -r requirements.txt`.

## Basic Usage

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(
        "tokenizer/zero_ne_tokenizer.json"
    )

    text = "नेपाल हाम्रो देश हो।"
    encoded = tokenizer.encode(text)

    print(encoded.tokens)
    print(encoded.ids)

    decoded = tokenizer.decode(encoded.ids)
    print("Round trip:", decoded == text)

## Run Tests

    python -m unittest discover -s tests -v

## Local API

Start the local development API with:

    python api/tokenize.py

Default address: `http://127.0.0.1:8766`

- GET `/health`
- POST `/api/tokenize`

The API returns token IDs, token pieces, offsets, Unicode grapheme information, and decoded text.

This API is intended for local research and development, not production deployment.

## Training

Training script: `tokenizer/train_ne_tokenizer.py`

The initial tokenizer was trained using processed Nepali Wikipedia content.

Expected training file:

`data/processed/wikipedia/ne/train.jsonl`

The original training corpus is not included in this repository.

## Dataset Attribution

The initial tokenizer was trained using processed Nepali Wikipedia content.

Wikipedia content is subject to applicable Creative Commons licensing and attribution requirements. Licensing and redistribution obligations for the trained tokenizer artifact require separate review.

## Research Status

This is an experimental research release. Further evaluation is needed for vocabulary coverage, segmentation quality, Unicode alignment, and downstream language-model performance.

## ZEROAILAB

Building language technology for Nepali and other underrepresented languages.

https://github.com/ZEROAILAB
