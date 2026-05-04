# Q4 errors hand-off (for Shoeb)

Source adapter: `HuggingFaceTB/SmolLM2-1.7B` + LoRA, trained on `en-UK` with seed 2024. Selected as the overall best adapter by mean validation Macro-F1 across the 3 test sets. Reported numbers and the misclassified examples below come from the held-out test split for that val-selected adapter.

The CSV `SmolLM2-1.7B_en-UK_seed2024_misclassified.csv` has 40 rows sampled (`random_state=42`) from all test-set misclassifications across en-UK, en-AU, and en-IN combined.

Columns:
- `text` -- the original review or comment
- `true_label` -- ground-truth Sarcasm label (0 = not_sarcastic, 1 = sarcastic)
- `predicted_label` -- model's prediction (same 0/1 encoding)
- `variety` -- which test set the example came from (en-UK / en-AU / en-IN)
- `base_model` -- Hugging Face ID of the base model used
- `training_variety` -- variety the adapter was trained on (en-UK in this case)
- `seed` -- random seed used during the adapter's training

For Q4: pick any 4 examples to write a Few-Shot Prompt with your own explanations, test the remaining 6 against that prompt per the spec, and report the before/after analysis.
