class PredictRequest(BaseModel):
    lr_results: dict
    bert_results: dict
    ollama_summary: str
"""    
backend/
│
├── src/
│    ├── app.py
│    ├── database/
│       ├── db.py
│       ├── models.py
│    
│    ├── routes/
│       ├── predict.py
│       ├── history.py
│    ├── schemas/
│       ├── request.py
│       ├── response.py
│
│    ├── services/
│       ├── bert_service.py
│       ├── history_service.py
│       ├── lr_service.py
│       ├── model_loader.py
│       ├── prompt_model.py
│
│    ├── utils/
│       ├── cleaner.py
│
├── trained/
│     ├── best_fine_tuned_bert/
│     ├── tfidf_vectorizer.pkl
│     ├── logistic_model.pkl
├── .dockerignore
├── .python-version
├── .pyproject.toml
├── .uv.lock
"""