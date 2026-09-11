"""
Prediction API Endpoints for AgriSense AI.
"""

import json
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import (
    PredictionInputSchema, BatchPredictionInputSchema,
    PredictionResponseSchema, FeedbackInputSchema
)
from app.services.prediction_service import run_prediction_service, run_batch_prediction_service
from app.models import Feedback

router = APIRouter(prefix="", tags=["Prediction Engine"])

@router.post("/predict", response_model=PredictionResponseSchema)
def predict_single_crop(input_data: PredictionInputSchema, db: Session = Depends(get_db)):
    """
    Predicts top crop recommendations based on farmer agricultural and environmental inputs.
    """
    inp_dict = input_data.model_dump()
    result = run_prediction_service(inp_dict, db)
    
    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Input Validation Failed", "errors": result.get("validation_errors")}
        )
        
    return result

@router.post("/predict/batch")
def predict_batch_crops(batch_data: BatchPredictionInputSchema, db: Session = Depends(get_db)):
    """
    Processes batch crop recommendations for multiple soil/farm samples.
    """
    inputs_list = [item.model_dump() for item in batch_data.inputs]
    results = run_batch_prediction_service(inputs_list, db)
    return {"total_processed": len(results), "results": results}

@router.post("/feedback")
def submit_prediction_feedback(feedback_data: FeedbackInputSchema, db: Session = Depends(get_db)):
    """
    Submits user satisfaction feedback for a recommendation.
    """
    record = Feedback(
        prediction_id=feedback_data.prediction_id,
        rating=feedback_data.rating,
        comments=feedback_data.comments
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"status": "success", "message": "Thank you for your feedback!", "feedback_id": record.id}
