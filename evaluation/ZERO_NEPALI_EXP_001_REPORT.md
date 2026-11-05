# PROJECT ZERO
## ZERO NEPALI EXP-001

Status: Completed

### Model
- Parameters: 98,130,432
- Vocabulary: 16,000
- Context length: 256
- Device: Apple MPS

### Training
- Training steps: 20
- Initial training loss: 9.755253
- Final training loss: 5.989724
- Best sampled validation loss: 5.917763

### Full Validation
- Validation tokens: 620,897
- Evaluated targets: 620,896
- Evaluation windows: 2,426
- Full validation loss: 5.976461
- Perplexity: 394.0434
- Evaluation duration: 65.43 seconds

### Checkpoint
checkpoints/nepali/zero_ne_exp001_best.pt

### Tokenizer
tokenizer/zero_ne_tokenizer.json

### Conclusion
The first Nepali training experiment completed successfully.
The checkpoint loads strictly and autoregressive inference runs.
Generated language quality remains poor.
Further training and evaluation are required.

### Next Experiment
ZERO NEPALI EXP-002
