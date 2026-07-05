from fastapi import APIRouter, HTTPException

from schemas.response import HistoryOverview, HistorySingle
from database.repository import get_all_predictions, get_prediction_by_id


# ========================== INITIALIZATION ==========================
router = APIRouter()

@router.get("/history", response_model=list[HistoryOverview])
def get_history():
    return (get_all_predictions())

@router.get("/history/{id}", response_model=HistorySingle)
def get_history_by_id(id: int):
    prediction = get_prediction_by_id(id)

    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found."
        )
    return prediction