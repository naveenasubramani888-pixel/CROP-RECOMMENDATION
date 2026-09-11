"""
Dashboard & Historical Analytics API Endpoints.
"""

import json
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import PredictionHistory
from app.services.recommendation_service import get_prediction_statistics

router = APIRouter(prefix="", tags=["Dashboard Analytics"])

@router.get("/statistics")
def get_dashboard_statistics(db: Session = Depends(get_db)):
    """
    Returns high-level platform summary statistics for the web dashboard.
    """
    return get_prediction_statistics(db)

@router.get("/history")
def get_prediction_history(limit: int = 50, db: Session = Depends(get_db)):
    """
    Retrieves historical logged predictions.
    """
    records = db.query(PredictionHistory).order_by(PredictionHistory.timestamp.desc()).limit(limit).all()
    
    history_list = []
    for r in records:
        history_list.append({
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "recommended_crop": r.recommended_crop,
            "confidence": r.confidence,
            "input_parameters": json.loads(r.input_parameters),
            "top_recommendations": json.loads(r.top_recommendations)
        })
        
    return {"total_records": len(history_list), "history": history_list}
