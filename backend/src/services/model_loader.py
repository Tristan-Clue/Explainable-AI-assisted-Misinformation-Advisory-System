import torch
import joblib

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

# Paths to trained models ## TO BE MOVED TO ENV
BERT_MODEL = "trained/best_fine_tuned_bert"
TFIDF_VECTOR = "trained/tfidf_vectorizer.pkl"
LR_MODEL = "trained/logistic_model.pkl"


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

# Initialize LIME explainer
lime_explainer = LimeTextExplainer(class_names=["Fake", "Real"])
