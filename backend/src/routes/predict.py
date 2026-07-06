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
        You will be given the output of two machine learning models that classified a news article as "fake" or "real". Your job is to explain their results in simple, human-readable language.

        Article:
        {text}

        Logistic Regression:
        Prediction: {lr_results.prediction}
        Fake Probability: {round(lr_results.probabilities.fake, 3)}
        Real Probability: {round(lr_results.probabilities.real, 3)}
        Words pushing Fake: {lr_results.explanations.push_fake}
        Words pushing Real: {lr_results.explanations.push_real}

        BERT:
        Prediction: {bert_results.prediction}
        Fake Probability: {round(bert_results.probabilities.fake, 3)}
        Real Probability: {round(bert_results.probabilities.real, 3)}

        STRICT GROUNDING RULES (most important):
        - Only state facts that are directly present in the data above. Do not invent reasoning, causes, or explanations that are not explicitly supported by the numbers or word lists given.
        - For Logistic Regression: you may only reference the exact words listed in "push_fake" and "push_real" as the reasons behind its prediction. Do not add outside interpretation of what those words "imply" about the article's content.
        - List up to 8 words for push_fake and up to 8 words for push_real, in the order given. If a list has fewer than 8 words, list all of them without inventing more.
        - Use clear transition phrases like "On the other hand" rather than "Conversely" when contrasting the two word lists.
        - For BERT: no word-level explanation data is provided. You must NOT guess, infer, or fabricate which words or phrases influenced BERT's decision. State only its prediction and probabilities, and explicitly note that no word-level explanation is available for BERT.
        - In the BERT section, note whether its fake probability is higher or lower than Logistic Regression's, and remind the reader that this confidence is not accompanied by any interpretable word-level reasoning, unlike Logistic Regression.

        CONFIDENCE BANDS:
        - Categorize each model's confidence (its probability for the predicted class) using these bands: 50-65% = "low confidence", 65-85% = "moderate confidence", 85-100% = "high confidence".
        - When both models agree, describe the combined signal as strong if both models are "moderate" or higher. Only describe the signal as weak or mixed if at least one model is in the "low confidence" band. Never describe agreement between two moderate-or-higher confidence models as a weak signal.
        - If the two models' fake probabilities differ by more than 15 percentage points, explicitly note this gap and what it suggests (e.g. one model is far more decisive than the other), without speculating on which model is "right".

        - In the Agreement section, only state whether the two predictions match or differ, and compare their confidence levels (probabilities) using the confidence bands above. Do not speculate about why they differ beyond the data given.

        - In the Conclusion, do NOT declare whether the article is actually fake or real. Instead, summarize:
        - How strong or weak the combined signal was, using the confidence bands above.
        - Whether the evidence came from concrete, interpretable signals (Logistic Regression's word-level reasons) or from a model with no explanation available (BERT).
        - What this means for how much a reader should trust this specific result (e.g. "this is a confidently-flagged case with interpretable reasoning" vs. "this is a borderline case where the models are hedging").
        - Keep this grounded strictly in the data provided — describe the pattern of the numbers, not the article's actual truthfulness.
        - Do NOT declare one model "more accurate," "more correct," or "more trustworthy" than the other anywhere in the response.
        - If something cannot be explained using only the given data, explicitly say "No explanation data is available for this," rather than filling the gap with speculation.

        OUTPUT FORMAT RULES:
        - Output ONLY plain Markdown text. Do not use any HTML tags (no <h1>, <p>, <ul>, <li>, etc.) and do not HTML-escape any characters (no &#39;, &quot;, etc.) — use normal characters like ' and " directly.
        - Do not wrap the output in a code block (no ``` fences).
        - Do not include any commentary about formatting or these instructions.
        - Use:
        - # for the main title
        - ## for section headers
        - **bold** for emphasis
        - - for bullet points

        Structure your response exactly like this:
        # Misinformation Analysis
        ## Article Prediction
        ## Logistic Regression
        ## BERT
        ## Agreement
        ## Conclusion
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
    insert_prediction(prediction)

    return prediction