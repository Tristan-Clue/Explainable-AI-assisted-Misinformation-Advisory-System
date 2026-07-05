from fastapi import APIRouter

from utils.cleaner import clean_text
from services.lr_service import lr_predict
from services.bert_service import bert_predict
from services.prompt_model import prompt_model

from schemas.request import InputRequest
from schemas.response import PredictResponse
from database.repository import insert_prediction
from config import OLLAMA_MODEL

# ========================== INITIALIZATION ==========================

router = APIRouter()

def build_prompt(text, lr_results, bert_results):
    return f"""
        You are an AI misinformation analysis assistant.

        Analyze the following predictions and explain them in simple language.

        Article:
        {text}

        Logistic Regression:
        Prediction: {lr_results.prediction}
        Fake Probability: {lr_results.probabilities.fake}
        Real Probability: {lr_results.probabilities.real}

        Words pushing Fake:
        {lr_results.explanations.push_fake}

        Words pushing Real:
        {lr_results.explanations.push_real}

        BERT:
        Prediction: {bert_results.prediction}
        Fake Probability: {bert_results.probabilities.fake}
        Real Probability: {bert_results.probabilities.real}

        Instructions:
        - Explain what both models think.
        - Mention if models agree or disagree.
        - Explain why in simple terms.
        - Do NOT invent information outside provided data.
        """


@router.post("/predict", response_model=PredictResponse)
def predict(request: InputRequest):
    text = request.text
    text = clean_text(text)
    lr_results = lr_predict(text)
    bert_results = bert_predict(text)
    ollama_summary = prompt_model(OLLAMA_MODEL, build_prompt(text, lr_results, bert_results))

    prediction =  PredictResponse(
        text=text,
        lr_results=lr_results,
        bert_results=bert_results,
        ollama_summary=ollama_summary
    )
    prediction_id = insert_prediction(prediction)

    return prediction