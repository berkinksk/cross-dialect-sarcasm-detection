import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel


def load(base_id, adapter_repo):
    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(
        base_id, num_labels=2, dtype=torch.bfloat16,
    )
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, adapter_repo)
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')
    return tok, model


@torch.no_grad()
def predict(tok, model, text, max_len=256):
    inputs = tok(text, return_tensors='pt', truncation=True, max_length=max_len, padding=True)
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    logits = model(**inputs).logits
    return int(logits.argmax(dim=-1))


if __name__ == "__main__":
    tok, model = load(
        "Qwen/Qwen2.5-1.5B",
        "berkinksk/besstie-sarcasm-en-UK-Qwen2.5-1.5B-seed2024",
    )
    text = "Oh great, another rainy day in Manchester. Just what I wanted."
    pred = predict(tok, model, text)
    print(pred)
