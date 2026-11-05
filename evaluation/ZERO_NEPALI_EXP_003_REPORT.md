# PROJECT ZERO
# ZERO NEPALI EXP-003 REPORT

## Experiment configuration

- Experiment: ZERO_NEPALI_EXP_003
- Source checkpoint: ZERO_NEPALI_EXP_002 best
- Architecture: ZEROTransformer
- Parameters: 98,130,432
- Vocabulary: 16,000
- Training steps: 5,000
- Learning rate: 0.00005
- Batch size: 1
- Sequence length: 256
- Optimizer: Fresh AdamW
- Device: Apple MPS

## Training results

- Final training loss: 2.8631
- Best sampled validation loss: 3.362318
- Training duration: 1,130.99 seconds

## Full validation results

- Checkpoint step: 5,000
- Validation tokens: 620,897
- Evaluated target tokens: 620,896
- Evaluation windows: 2,426
- Full validation loss: 3.250068
- Perplexity: 25.7921
- Evaluation duration: 66.32 seconds

## Historical comparison

| Experiment | Full validation loss | Perplexity |
|---|---:|---:|
| EXP-001 | 5.976461 | 394.0434 |
| EXP-002 | 3.878825 | 48.3673 |
| EXP-003 | 3.250068 | 25.7921 |

## Methodology

EXP-003 started from the best EXP-002 model weights.
A fresh AdamW optimizer was used.

Full evaluation used the same validation token stream,
256-token windows, batch size 4, and masked padding
procedure as EXP-002.

## Limitations

This evaluation measures next-token prediction loss.
It does not independently establish language fluency,
factual accuracy, reasoning, or conversational ability.

The validation stream is evaluated in sequential chunks.
Document boundaries are not independently preserved
by this evaluation procedure.

## Checkpoint

checkpoints/nepali_exp_003/zero_ne_exp003_best.pt

## Status

Training completed.
Full validation completed.
Generation quality evaluation pending.
