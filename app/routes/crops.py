"""
Crop Knowledge Base & Model Info API Endpoints.
"""

from fastapi import APIRouter, HTTPException, status
from app.services.recommendation_service import get_all_crop_information
from app.services.explanation_service import get_model_insights_service

router = APIRouter(prefix="", tags=["Crop Knowledge & Model Info"])

@router.get("/crops")
def list_all_crops():
    """
    Returns agronomic knowledge base guidelines for all 20 crops.
    """
    kb = get_all_crop_information()
    return {"total_crops": len(kb), "crops": kb}

@router.get("/crops/{crop_name}")
def get_crop_details(crop_name: str):
    """
    Returns specific agronomic growing guidelines for a given crop name.
    """
    kb = get_all_crop_information()
    # Case-insensitive lookup
    matched_crop = next((c for c in kb.keys() if c.lower() == crop_name.lower()), None)
    
    if not matched_crop:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Crop '{crop_name}' not found in knowledge base."
        )
        
    return {"crop": matched_crop, "details": kb[matched_crop]}

@router.get("/model-info")
def get_model_information():
    """
    Returns model training metrics, comparative algorithm scores, and baseline info.
    """
    return get_model_insights_service()
