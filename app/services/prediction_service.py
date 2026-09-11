"""
Prediction service interfacing between API routes, ML prediction pipeline, and SQLite DB.
"""

import json
from sqlalchemy.orm import Session
from typing import Dict, Any, List
from src.predict import predict_crop
from app.models import PredictionHistory

def run_prediction_service(input_data: Dict[str, Any], db: Session = None) -> Dict[str, Any]:
    """Runs crop prediction and logs result in SQLite database if db session provided."""
    result = predict_crop(input_data)
    
    if result.get("success") and db is not None:
        try:
            history_record = PredictionHistory(
                input_parameters=json.dumps(input_data),
                recommended_crop=result["recommended_crop"],
                confidence=result["confidence"],
                top_recommendations=json.dumps(result["top_3_recommendations"])
            )
            db.add(history_record)
            db.commit()
            db.refresh(history_record)
            result["prediction_id"] = history_record.id
        except Exception as e:
            db.rollback()
            
    return result

def run_batch_prediction_service(inputs: List[Dict[str, Any]], db: Session = None) -> List[Dict[str, Any]]:
    """Runs batch predictions for multiple input samples."""
    results = []
    for inp in inputs:
        results.append(run_prediction_service(inp, db))
    return results
