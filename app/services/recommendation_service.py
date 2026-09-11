"""
Recommendation service for fetching crop knowledge base and statistics.
"""

from typing import Dict, Any, List
from sqlalchemy.orm import Session
from src.utils import CROP_KNOWLEDGE_BASE
from app.models import PredictionHistory

def get_all_crop_information() -> Dict[str, Any]:
    """Returns knowledge base of all 20 crops."""
    return CROP_KNOWLEDGE_BASE

def get_prediction_statistics(db: Session) -> Dict[str, Any]:
    """Returns aggregated summary statistics of predictions from SQLite DB."""
    records = db.query(PredictionHistory).all()
    total_count = len(records)
    
    if total_count == 0:
        return {
            "total_predictions": 0,
            "most_recommended_crop": "None",
            "average_confidence": 0.0,
            "crop_distribution": {}
        }
        
    crop_counts = {}
    total_conf = 0.0
    
    for r in records:
        crop_counts[r.recommended_crop] = crop_counts.get(r.recommended_crop, 0) + 1
        total_conf += r.confidence
        
    most_recommended = max(crop_counts, key=crop_counts.get)
    avg_conf = round(total_conf / total_count, 2)
    
    return {
        "total_predictions": total_count,
        "most_recommended_crop": most_recommended,
        "average_confidence": avg_conf,
        "crop_distribution": crop_counts
    }
