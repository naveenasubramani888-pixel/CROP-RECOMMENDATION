"""
Explanation service providing model insight metrics and parameters.
"""

import pandas as pd
from typing import Dict, Any
from src.utils import MODEL_COMPARISON_PATH

def get_model_insights_service() -> Dict[str, Any]:
    """Returns stored model comparison performance metrics."""
    if MODEL_COMPARISON_PATH.exists():
        df_comp = pd.read_csv(MODEL_COMPARISON_PATH)
        best_model = df_comp.iloc[0].to_dict()
        comparison_list = df_comp.to_dict(orient="records")
    else:
        best_model = {}
        comparison_list = []
        
    return {
        "best_model": best_model,
        "all_models_comparison": comparison_list
    }
