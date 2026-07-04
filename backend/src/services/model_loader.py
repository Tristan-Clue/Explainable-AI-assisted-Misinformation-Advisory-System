import torch
import joblib

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import TRAINED_DIR

BERT_MODEL = f"{TRAINED_DIR}/best_fine_tuned_bert"
TFIDF_VECTOR = f"{TRAINED_DIR}/tfidf_vectorizer.pkl"
LR_MODEL = f"{TRAINED_DIR}/logistic_model.pkl"

# Check GPU for BERT model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load Logistic Regression and TF-IDF models
tfidf = joblib.load(TFIDF_VECTOR)
lr_model = joblib.load(LR_MODEL)

# Load BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL)

# Move model to the appropriate device and set it to evaluation mode
model.to(device)
model.eval()
