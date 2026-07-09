from fastapi import APIRouter, HTTPException

from utils.cleaner import clean_text
from services.lr_service import lr_predict
from services.bert_service import bert_predict
from services.prompt_model import prompt_model

from schemas.request import InputRequest
from schemas.response import PredictResponse
from database.repository import insert_prediction
from config import OLLAMA_MODEL
from config import GEMINI_MODEL
from utils.verify import verify_medical

# ========================== INITIALIZATION ==========================

router = APIRouter()

def build_prompt(text, lr_results, bert_results):
    return f"""
        You are an AI assistant that interprets the outputs of multiple machine learning models used to detect medical misinformation.

        You are given:
        - The original article.
        - A Logistic Regression prediction with probabilities and word-level explanation scores.
        - A BERT prediction with probabilities.

        Your role is to explain these results in a clear, balanced, and educational way for non-technical readers. Your goal is to help readers understand both the article and the model predictions, without determining whether the article is actually true or false.

        Article:
        {text}

        Logistic Regression:
        Prediction: {lr_results.prediction}
        Fake Probability: {round(lr_results.probabilities.fake, 3)}
        Real Probability: {round(lr_results.probabilities.real, 3)}
        Words pushing Fake:
        {lr_results.explanations.push_fake}

        Words pushing Real:
        {lr_results.explanations.push_real}

        BERT:
        Prediction: {bert_results.prediction}
        Fake Probability: {round(bert_results.probabilities.fake, 3)}
        Real Probability: {round(bert_results.probabilities.real, 3)}

        ----------------------------
        GENERAL RESPONSIBILITIES
        ----------------------------

        Write an informative analysis that explains the machine learning predictions while helping the reader understand the article.

        Use clear, natural language suitable for someone without a machine learning background.

        Keep the overall response between approximately 400 and 600 words.

        Do not simply repeat the model outputs. Instead, explain what the probabilities, confidence levels, and prediction agreement mean.

        ----------------------------
        ARTICLE OVERVIEW
        ----------------------------

        Begin by briefly summarizing the article in two or three concise paragraphs.

        If the article discusses scientific or medical research, briefly explain the type of research involved (for example, disease mechanism research, laboratory study, clinical trial, observational study, treatment research, or expert opinion).

        You may use general medical knowledge to explain research concepts so the article is easier to understand.

        Keep these explanations brief.

        Do NOT determine whether the research itself is correct or incorrect.

        Do NOT fact-check the article.

        ----------------------------
        OVERALL ASSESSMENT
        ----------------------------

        Summarize the overall prediction from both models.

        Explain:

        - whether both models agree or disagree
        - which model is more confident
        - whether the overall signal appears consistent

        Do not declare the article to be true or false.

        ----------------------------
        LOGISTIC REGRESSION ANALYSIS
        ----------------------------

        Explain:

        - the predicted class
        - the prediction probability
        - the confidence level

        Use these confidence bands:

        50–65% = Low confidence

        65–75% = Moderate confidence

        75-90% = High confidence

        90–100% = Very high confidence

        Briefly explain what the confidence means in plain language.

        List up to the top eight words contributing toward Fake.

        List up to the top eight words contributing toward Real.

        Then explain that:

        - these scores represent statistical patterns learned during training
        - larger magnitude scores indicate stronger influence on the prediction
        - the scores do not explain why the model learned those associations
        - individual words alone do not determine whether an article is misinformation

        You may explain the meaning of important medical or scientific terms if they help readers understand the article, but do not use those explanations to justify the Logistic Regression prediction.

        Never invent reasons why a specific word received its score.

        ----------------------------
        BERT ANALYSIS
        ----------------------------

        Explain:

        - the predicted class
        - the prediction probability
        - the confidence level

        Briefly explain what the confidence means.

        Compare BERT's confidence with Logistic Regression.

        State that BERT provides only prediction probabilities and does not provide interpretable word-level contribution scores.

        Do not speculate about which words influenced BERT.

        ----------------------------
        AGREEMENT
        ----------------------------

        State whether both models agree or disagree.

        If they agree:

        Summarize the combined prediction and compare their confidence levels.

        If they disagree:

        Summarize both predictions and explain that the models reached different conclusions.

        If their confidence differs by more than 15 percentage points, mention this difference.

        Do not speculate about why they disagree.

        ----------------------------
        CONCLUSION
        ----------------------------

        Provide a concise overall interpretation of the model outputs.

        Summarize:

        - the overall prediction
        - how confident the combined result appears
        - whether both models produced a consistent assessment

        Keep the conclusion focused on interpreting the machine learning outputs.

        Do not introduce new information.

        Do not state or imply that the article is objectively true or false.

        ----------------------------
        STRICT GROUNDING RULES
        ----------------------------

        You may use general knowledge only to explain:

        - medical terminology
        - research concepts
        - machine learning concepts
        - confidence scores

        You must NOT:

        - invent evidence
        - fabricate reasoning for either model
        - explain why Logistic Regression learned a particular word weight
        - speculate about BERT's internal reasoning
        - claim that the article is true or false
        - claim one model is more accurate or trustworthy than the other

        If something cannot be explained using the provided data, simply state that no further explanation is available.

        ----------------------------
        OUTPUT FORMAT
        ----------------------------

        Output ONLY Markdown.

        Do not output HTML.

        Do not wrap the response in a code block.

        Use the following structure exactly:

        # Misinformation Analysis

        ## Article Overview

        ## Overall Assessment

        ## Logistic Regression Analysis

        ## BERT Analysis

        ## Agreement

        ## Conclusion
        """


@router.post("/predict", response_model=PredictResponse)
def predict(request: InputRequest):
    text = request.text
    if not verify_medical(text):
        raise HTTPException(
            status_code=422,
            detail="The submitted text does not appear to be a medical news article."
        )
    text = clean_text(text)
    lr_results = lr_predict(text)
    bert_results = bert_predict(text)
    ollama_summary = prompt_model(GEMINI_MODEL, build_prompt(text, lr_results, bert_results))

    prediction =  PredictResponse(
        text=text,
        lr_results=lr_results,
        bert_results=bert_results,
        ollama_summary=ollama_summary
    )
    insert_prediction(prediction)

    return prediction