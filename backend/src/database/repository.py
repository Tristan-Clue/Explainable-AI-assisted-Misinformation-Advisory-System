import json

from datetime import datetime

from database.db import get_connection
from schemas.response import PredictResponse, HistoryOverview, HistorySingle, LRResult, LRExplanations, ExplanationWord, ProbabilityOutput, BERTResult

def insert_prediction(prediction:  PredictResponse) -> int :
    conn = get_connection()

    try:
        cur = conn.cursor()
        cur.execute("""           
            INSERT INTO history (
                    text,
                    lr_prediction,
                    lr_fake_prob,
                    lr_real_prob,
                    lr_explanations,
                    bert_prediction,
                    bert_fake_prob,
                    bert_real_prob,
                    ollama_summary,
                    created_at
                    )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (prediction.text, prediction.lr_results.prediction,
             prediction.lr_results.probabilities.fake, prediction.lr_results.probabilities.real,
             json.dumps(prediction.lr_results.explanations.model_dump()),
             prediction.bert_results.prediction, prediction.bert_results.probabilities.fake,
             prediction.bert_results.probabilities.real, prediction.ollama_summary,
             datetime.now()))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()

def get_all_predictions() -> list[HistoryOverview]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""           
            SELECT
                id,
                text,
                lr_prediction,
                bert_prediction,
                created_at
                FROM history
                ORDER BY created_at DESC;
            """)
        rows = cur.fetchall()

        history = []

        for row in rows:
            preview = row["text"][:100]

            if len(row["text"]) > 100:
                preview += "..."
            history.append(
                HistoryOverview(
                    id=row["id"],
                    preview=preview,      # first 100 characters
                    lr_prediction=row["lr_prediction"],
                    bert_prediction=row["bert_prediction"],
                    created_at=row["created_at"]
                )
            )

        return history
    
    finally:
        conn.close()

def get_prediction_by_id(id: int) -> HistorySingle | None:
    conn = get_connection()

    try:
        cur = conn.cursor()

        cur.execute("""
            SELECT
                id, text, lr_prediction, lr_fake_prob, lr_real_prob, lr_explanations,
                bert_prediction, bert_fake_prob, bert_real_prob, ollama_summary, created_at
            FROM history
            WHERE id = ?;
        """, (id,))

        row = cur.fetchone()

        if row is None:
            return None

        explanations_json = json.loads(row["lr_explanations"])

        # Rebuilding JSON format back to LRExplanation Class
        lr_explanations = LRExplanations(
            push_fake=[
                ExplanationWord(**item)
                for item in explanations_json["push_fake"]
            ],
            push_real=[
                ExplanationWord(**item)
                for item in explanations_json["push_real"]
            ]
        )

        # Rebuilding JSON format back to LRResult Class
        lr_result = LRResult(
            prediction=row["lr_prediction"],
            prediction_id=1 if row["lr_prediction"] == "Real" else 0,
            probabilities=ProbabilityOutput(
                fake=row["lr_fake_prob"],
                real=row["lr_real_prob"]
            ),
            explanations=lr_explanations
        )

        # Rebuilding JSON format back to BERTResult Class
        bert_result = BERTResult(
            prediction=row["bert_prediction"],
            prediction_id=1 if row["bert_prediction"] == "Real" else 0,
            probabilities=ProbabilityOutput(
                fake=row["bert_fake_prob"],
                real=row["bert_real_prob"]
            )
        )

        return HistorySingle(
            id=row["id"],
            created_at=row["created_at"],
            text=row["text"],
            lr_results=lr_result,
            bert_results=bert_result,
            ollama_summary=row["ollama_summary"]
        )

    finally:
        conn.close()
