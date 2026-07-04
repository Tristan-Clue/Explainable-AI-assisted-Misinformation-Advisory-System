import re

def clean_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\s*\S+|www\.\S+', '', text)

    # Remove escaped unicode / weird space
    text = text.replace("\\xa0", " ")
    text = text.replace("\xa0", " ")

    # Remove escaped newlines
    text = text.replace("\\n", " ")
    text = text.replace("\n", " ")

    # Remove dict symbols / punctuation junk
    text = re.sub(r"[{}\[\]\"']", " ", text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
