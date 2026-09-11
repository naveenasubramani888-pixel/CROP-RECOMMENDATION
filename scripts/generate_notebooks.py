"""
Script to create the 5 project Jupyter Notebooks (.ipynb) in notebooks/.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)

def create_notebook(filename: str, cells_data: list):
    cells = []
    for cell_type, source in cells_data:
        cells.append({
            "cell_type": cell_type,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")]
        })
        
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.11.7"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    path = NOTEBOOKS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {path}")

# Notebook 1
nb1 = [
    ("markdown", "# 01 - Synthetic Agricultural Data Generation\n\nThis notebook demonstrates domain-informed synthetic dataset generation for AgriSense AI."),
    ("code", "import sys\nfrom pathlib import Path\nsys.path.append('../')\nfrom src.data_generation import generate_agricultural_dataset, save_raw_dataset\nfrom src.utils import RAW_DATA_PATH"),
    ("code", "df = generate_agricultural_dataset(num_samples=15000, random_seed=42)\ndf.head()"),
    ("code", "print('Shape:', df.shape)\nprint('Crops:', df['crop'].value_counts())")
]

# Notebook 2
nb2 = [
    ("markdown", "# 02 - Data Cleaning & Preprocessing Pipeline\n\nHandles missing values, duplicate removal, invalid bounds filtering, and scikit-learn ColumnTransformer pipeline fitting."),
    ("code", "import sys\nimport pandas as pd\nsys.path.append('../')\nfrom src.data_preprocessing import clean_data, preprocess_and_split_data\nfrom src.utils import RAW_DATA_PATH"),
    ("code", "raw_df = pd.read_csv(RAW_DATA_PATH)\ncleaned_df, audit = clean_data(raw_df)\nprint(audit)"),
    ("code", "X_tr, X_te, y_tr, y_te, pipe, le = preprocess_and_split_data(cleaned_df)\nprint('Processed Train Shape:', X_tr.shape)")
]

# Notebook 3
nb3 = [
    ("markdown", "# 03 - Exploratory Data Analysis (EDA)\n\nVisualizing soil nutrient distributions, environmental metrics, correlation matrices, and crop suitability ranges."),
    ("code", "import sys\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nsys.path.append('../')\nfrom src.utils import CLEANED_DATA_PATH"),
    ("code", "df = pd.read_csv(CLEANED_DATA_PATH)\ndf.info()"),
    ("code", "df.describe()"),
    ("code", "plt.figure(figsize=(10,6))\nsns.histplot(df['rainfall'], kde=True)\nplt.title('Rainfall Distribution')\nplt.show()")
]

# Notebook 4
nb4 = [
    ("markdown", "# 04 - Multi-Algorithm Model Training & Tuning\n\nTraining 10 ML models including Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Gradient Boosting, Extra Trees, XGBoost, and Ensembles."),
    ("code", "import sys\nimport numpy as np\nsys.path.append('../')\nfrom src.train import train_all_models\nfrom src.utils import X_TRAIN_PATH, Y_TRAIN_PATH"),
    ("code", "X_train = np.loadtxt(X_TRAIN_PATH, delimiter=',')\ny_train = pd.read_csv(Y_TRAIN_PATH)['encoded'].values\nprint('Loaded train data:', X_train.shape)")
]

# Notebook 5
nb5 = [
    ("markdown", "# 05 - Model Evaluation, Explainability & Selection\n\nComparing Macro F1, Accuracy, Precision, Recall, CV scores, confusion matrix, and feature importances."),
    ("code", "import sys\nimport pandas as pd\nsys.path.append('../')\nfrom src.utils import MODEL_COMPARISON_PATH"),
    ("code", "comparison_df = pd.read_csv(MODEL_COMPARISON_PATH)\ncomparison_df")
]

def main():
    create_notebook("01_data_generation.ipynb", nb1)
    create_notebook("02_data_cleaning.ipynb", nb2)
    create_notebook("03_eda.ipynb", nb3)
    create_notebook("04_model_training.ipynb", nb4)
    create_notebook("05_model_evaluation.ipynb", nb5)

if __name__ == "__main__":
    main()
