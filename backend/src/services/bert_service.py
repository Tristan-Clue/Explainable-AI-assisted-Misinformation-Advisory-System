import torch

from services.model_loader import (tokenizer, model, device)
from schemas.response import BERTResult, ProbabilityOutput

def bert_predict(texts):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    bert_pred = torch.argmax(logits, dim=1).item()
    bert_probs = torch.softmax(logits, dim=1)

    return BERTResult(
        prediction="Real" if bert_pred == 0 else "Fake",
        prediction_id=bert_pred,
        probabilities=ProbabilityOutput(
            real=float(bert_probs[0][0].item()),
            fake=float(bert_probs[0][1].item())
        )
    )
