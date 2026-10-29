
import sys
from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_validate,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ==========================================
# 1. PROJECT PATH AND IMPORTS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow imports from src
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing.classification_loader import load_data
from classification.split_data import split_classification_data
from classification.preprocessing import build_preprocessor


# ==========================================
# 2. TRAIN BASELINE MODEL
# ==========================================

def train_baseline():

    # Step 1: Load dataset
    print("\n===== LOADING DATA =====")

    df = load_data()

    print("Dataset shape:", df.shape)

    # Step 2: Split dataset
    X_train, X_test, y_train, y_test = (
        split_classification_data(df)
    )

    # Step 3: Create preprocessing + model pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            ),
        ]
    )

    # ==========================================
    # 3. TRAIN MODEL
    # ==========================================

    print("\n===== TRAINING BASELINE MODEL =====")

    model.fit(X_train, y_train)

    print("Model trained successfully!")

    # ==========================================
    # 4. TRAINING PREDICTIONS
    # ==========================================

    y_pred = model.predict(X_train)

    # ==========================================
    # 5. TRAINING METRICS
    # ==========================================

    print("\n===== BASELINE TRAINING METRICS =====")

    print(
        "Accuracy:",
        round(accuracy_score(y_train, y_pred), 4)
    )

    print(
        "Precision:",
        round(precision_score(y_train, y_pred), 4)
    )

    print(
        "Recall:",
        round(recall_score(y_train, y_pred), 4)
    )

    print(
        "F1 Score:",
        round(f1_score(y_train, y_pred), 4)
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_train, y_pred))

    # ==========================================
    # 6. GROUPED CROSS-VALIDATION
    # ==========================================

    print("\n===== 5-FOLD CROSS-VALIDATION =====")

    # Identical predictor rows receive the same group
    groups = pd.util.hash_pandas_object(
        X_train,
        index=False
    )

    # Keep identical feature groups together
    # while attempting to balance target classes
    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Train and validate five times
    scores = cross_validate(
        model,
        X_train,
        y_train,
        groups=groups,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",
        },
        n_jobs=1,
    )

    # ==========================================
    # 7. DISPLAY CROSS-VALIDATION RESULTS
    # ==========================================

    print("\n===== CROSS-VALIDATION RESULTS =====")

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]:

        values = scores[f"test_{metric}"]

        print(
            f"{metric}: "
            f"{values.mean():.4f} "
            f"(std: {values.std():.4f})"
        )

    print("\nCross-validation completed successfully!")

    # Return model for future use
    return model


# ==========================================
# 8. RUN SCRIPT
# ==========================================

if __name__ == "__main__":
    train_baseline()
