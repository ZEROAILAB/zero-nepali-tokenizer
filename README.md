# ZERO Nepali Tokenizer

Developed by ZEROAILAB (https://github.com/ZEROAILAB)

Experimental tokenizer research for Nepali and other underrepresented languages.

## Releases and Artifacts

| Release | Training Data | Vocabulary | Status |
|---|---|---:|---|
| ZERO Nepali Tokenizer v0.1 | Processed Nepali Wikipedia | 16,000 | Experimental Nepali-first tokenizer |
| ZERO Tokenizer v0.2 | TinyStories English | 12,710 | Experimental ByteLevel BPE tokenizer |

## v0.1: Nepali-first Tokenizer

ZERO v0.1 is an experimental Nepali-first Byte-Level BPE tokenizer.

**Configuration**

- Algorithm: Byte Pair Encoding (BPE)
- Pre-tokenizer: ByteLevel
- Decoder: ByteLevel
- Normalization: Unicode NFC
- Requested vocabulary size: 16,000
- Special tokens: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`

**Training data**

Processed Nepali Wikipedia content.

Expected training file:

`data/processed/wikipedia/ne/train.jsonl`

The training script reads JSONL records with `title` and `text` fields.

**Training script**

`tokenizer/train_ne_tokenizer.py`

**Model artifact**

`tokenizer/zero_ne_tokenizer.json`

The processed training corpus is not included in this repository.

## v0.2: ZERO Tokenizer

ZERO v0.2 is an experimental ByteLevel BPE tokenizer trained using the TinyStories English dataset.

**Configuration**

- Algorithm: Byte Pair Encoding (BPE)
- Pre-tokenizer: ByteLevel
- Decoder: ByteLevel
- Normalization: Unicode NFC
- Initial alphabet: ByteLevel alphabet
- Requested vocabulary size: 16,000
- Achieved vocabulary size: 12,710
- Minimum token frequency: 2
- Special tokens: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`

**Dataset**

- Hugging Face dataset: `roneneldan/TinyStories`
- Split: `train[:10000]`
- Download script: `data/download_dataset.py`
- Local dataset output: `data/raw/tinystories.txt`

**Training script**

`tokenizer/train_tokenizer_v02.py`

**Model artifact**

`tokenizer/zero_tokenizer_v02.json`

**Important limitation:** v0.2 was trained on English TinyStories data. It is not a Nepali-trained tokenizer. Unicode round-trip tests demonstrate text preservation for the tested samples, not Nepali vocabulary optimization or linguistic quality.

## Installation

Requirements:

- Python 3.9 or newer
- Git

Clone the repository:

```bash
git clone https://github.com/ZEROAILAB/zero-nepali-tokenizer.git
cd zero-nepali-tokenizer
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Usage

### Load v0.1

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    "tokenizer/zero_ne_tokenizer.json"
)

text = "नेपाल हाम्रो देश हो।"
encoded = tokenizer.encode(text)

print("Tokens:", encoded.tokens)
print("Token IDs:", encoded.ids)

decoded = tokenizer.decode(encoded.ids)
print("Decoded:", decoded)
```

### Load v0.2

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    "tokenizer/zero_tokenizer_v02.json"
)

text = "Artificial intelligence and language technology."
encoded = tokenizer.encode(text)

print("Tokens:", encoded.tokens)
print("Token IDs:", encoded.ids)

decoded = tokenizer.decode(encoded.ids)
print("Decoded:", decoded)
```

## Reproduce v0.2 Training

Download the dataset:

```bash
python data/download_dataset.py
```

Train the tokenizer:

```bash
python tokenizer/train_tokenizer_v02.py
```

The dataset is saved locally at `data/raw/tinystories.txt`. The trained tokenizer is saved at `tokenizer/zero_tokenizer_v02.json`. The dataset is not committed to this repository.

## Testing

Run the complete test suite:

```bash
python -m unittest discover -s tests -v
```

The current test suite contains six tests covering:

- v0.1 Nepali text round-trip
- v0.1 special token IDs
- v0.1 vocabulary size
- v0.2 text round-trip
- v0.2 special token IDs
- v0.2 vocabulary size

Latest locally reported result: **6 tests passed**. These tests verify basic tokenizer behavior, not downstream language-model performance.

## Local API

The local development API loads the v0.1 Nepali tokenizer.

Start the server:

```bash
python api/tokenize.py
```

Default address: `http://127.0.0.1:8766`

### Health check

```bash
curl http://127.0.0.1:8766/health
```

### Tokenization request

```bash
curl -X POST http://127.0.0.1:8766/api/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text":"नेपाल हाम्रो देश हो।"}'
```

The API returns the input text, token pieces and IDs, token offsets, Unicode grapheme information, decoded text, round-trip comparison, vocabulary size, and special token IDs.

This API is intended for local research and development, not production deployment.

## Repository Structure

```text
zero-nepali-tokenizer/
├── api/
│   └── tokenize.py
├── data/
│   └── download_dataset.py
├── evaluation/
├── inference/
├── model/
├── tests/
│   ├── test_tokenizer.py
│   └── test_tokenizer_v02.py
├── tokenizer/
│   ├── train_ne_tokenizer.py
│   ├── train_tokenizer.py
│   ├── train_tokenizer_v02.py
│   ├── zero_ne_tokenizer.json
│   ├── zero_tokenizer.json
│   └── zero_tokenizer_v02.json
├── training/
├── generate.py
├── requirements.txt
└── README.md
```

## Dependencies

```text
tokenizers==0.22.2
regex==2026.1.15
datasets==5.2.0
```

## Research Status

ZERO Nepali Tokenizer is an ongoing language technology research project.

Current work focuses on:

- Nepali tokenization
- Unicode text handling
- Byte-Level BPE experiments
- Dataset preparation
- Tokenizer evaluation
- Language technology for underrepresented languages

Further evaluation is required for vocabulary coverage, segmentation quality, linguistic performance, and downstream language-model applications.

## Dataset Attribution

The v0.1 tokenizer was trained using processed Nepali Wikipedia content. Wikipedia content is subject to applicable Creative Commons licensing and attribution requirements. Licensing and redistribution obligations for the trained tokenizer artifact require separate review.

The v0.2 tokenizer was trained using the `roneneldan/TinyStories` dataset. Review the dataset's source page and license before redistributing dataset content or derived materials.

## ZEROAILAB

Building language technology for Nepali and other underrepresented languages.

https://github.com/ZEROAILAB
