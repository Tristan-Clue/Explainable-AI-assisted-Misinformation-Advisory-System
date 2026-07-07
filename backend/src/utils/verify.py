from services.prompt_model import prompt_model
from config import MODEL

keywords = ['medical']

def verify_medical(text:str):

    prompt = f"""
    You are a classifier that determines whether a given input is a medical news article.

    Input:
    {text}
    
    Instructions:
    - Determine whether the input above is a medical news article
    - The input must be a full article (multiple sentences forming a coherent piece of writing with a topic, context, and narrative). If the input is just a single sentence, a phrase, a headline only, or a fragment with no article-like structure, answer No
    - A medical news article discusses topics such as diseases, treatments, drugs, medical research, public health, healthcare policy, hospitals, clinical trials, or similar health-related subject matter, presented in a news or journalistic format
    - Respond with exactly one word: Yes or No
    - Do not include punctuation, explanations, or any other text
    - Do not use markdown, quotes, or formatting of any kind
    """
    reply = prompt_model(MODEL, prompt)
    if reply.lower() == "yes":
        return True
    else:
        return False