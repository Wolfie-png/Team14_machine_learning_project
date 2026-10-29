
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Locate the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow imports from the src folder
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing.classification_loader import load_data
from classification.features import FEATURES, TARGET


def split_classification_data(df):

    # Separate predictors and target
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Assign the same group ID to identical feature rows
    group_ids = pd.util.hash_pandas_object(
        X,
        index=False
    )

    # Split entire groups, not individual rows
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=group_ids)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print("\n===== CLASSIFICATION SPLIT =====")

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    print("\nTraining cancellation rate:")
    print(y_train.mean() * 100)

    print("\nTesting cancellation rate:")
    print(y_test.mean() * 100)

    # Verify that no feature group appears in both sets
    train_groups = set(group_ids.iloc[train_idx])
    test_groups = set(group_ids.iloc[test_idx])

    overlap = train_groups.intersection(test_groups)

    print("\nOverlapping feature groups:", len(overlap))

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    df = load_data()

    X_train, X_test, y_train, y_test = (
        split_classification_data(df)
    )
