# BESSTIE Sarcasm LoRA -- en-IN

- Base: `Qwen/Qwen2.5-1.5B`
- Training data: BESSTIE `en-IN` train split (seed 2024, 3 epochs, class-weighted CE, LoRA r=8 alpha=16 dropout=0.1 on q_proj+v_proj)
- Dataset: `surrey-nlp/BESSTIE-CW-26` (Hugging Face)
- Adapter selection: validation split (mean Macro-F1 across 3 seeds); reporting on held-out test split
- Test Macro-F1: **0.6019** (this checkpoint), 0.6078 (3-seed mean)

```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
base = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen2.5-1.5B", num_labels=2)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
model = PeftModel.from_pretrained(base, "berkinksk/besstie-sarcasm-en-IN-Qwen2.5-1.5B-seed2024")
```

## Limitations and notes

- Validation sets per variety are small (95-117 examples), so adapter selection on validation is noisy. The 3-seed-mean test macro-F1 is the more reliable performance number.
- Sarcasm class is imbalanced in BESSTIE; F1 on the sarcastic class is more informative than macro-F1 alone.
- Trained for an MSc coursework project; not intended as a production model.

## License

Research use only. Base model weights are governed by the base model's original license (see `Qwen/Qwen2.5-1.5B` on Hugging Face). Adapter weights follow the same restriction as the base.
