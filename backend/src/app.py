from fastapi import FastAPI
from routes.predict import router as predict_router
from routes.history import router as history_router
from routes.history import router as historyid_router
from database.db import init_db

app = FastAPI(title="Medical Fake News Detection API")

# Initialize DB
init_db()

# Attach routes from predict.py and history.py to the main FastAPI app
app.include_router(predict_router)
app.include_router(history_router)
app.include_router(historyid_router)