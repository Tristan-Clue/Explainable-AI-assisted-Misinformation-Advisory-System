import torch

from src.services.model_loader import (tokenizer, model, device, lime_explainer)
from src.schemas.response import BERTResult, ProbabilityOutput, BERTExplanations, ExplanationWord

# Used by LIME for explanation
def bert_predict_proba(texts):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        ).cpu().numpy()

    return probabilities


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

    explanations = lime_explainer.explain_instance(
        texts,
        bert_predict_proba,
        num_features=10,
        labels=[bert_pred]
    )

    # Get Lime explanation for BERT prediction
    lime_results = explanations.as_list(label=bert_pred)
    lime_explanations = [{"word": word, "score": score} for word, score in lime_results]
    
    # Splitting positive and negative contributions
    push_positive = [x for x in lime_explanations if x["score"] > 0]
    push_negative = [x for x in lime_explanations if x["score"] < 0]

    return BERTResult(
        prediction="Real" if bert_pred == 1 else "Fake",
        prediction_id=bert_pred,
        probabilities=ProbabilityOutput(
            fake=float(bert_probs[0][0].item()),
            real=float(bert_probs[0][1].item())
        ),
        explanations=BERTExplanations(
            support_prediction=push_positive,
            against_prediction=push_negative
        )
    )
