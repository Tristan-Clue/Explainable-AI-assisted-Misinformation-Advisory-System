from pydantic import BaseModel


class ProbabilityOutput(BaseModel):
    fake: float
    real: float


class ExplanationWord(BaseModel):
    word: str
    score: float


class LRExplanations(BaseModel):
    push_fake: list[ExplanationWord]
    push_real: list[ExplanationWord]


class LRResult(BaseModel):
    prediction: str
    prediction_id: int
    probabilities: ProbabilityOutput
    explanations: LRExplanations


class BERTResult(BaseModel):
    prediction: str
    prediction_id: int
    probabilities: ProbabilityOutput


class PredictResponse(BaseModel):
    text: str
    lr_results: LRResult
    bert_results: BERTResult
    ollama_summary: str