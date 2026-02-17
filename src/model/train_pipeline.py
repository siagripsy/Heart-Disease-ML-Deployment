from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    classification_report,
    recall_score,
    roc_auc_score,
)

from sklearn.utils.class_weight import compute_sample_weight

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


DATA_PATH = Path("data/processed/heart_clean.csv")
ARTIFACT_DIR = Path("artifacts/models")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def split_features(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_features = X.columns.tolist()
    categorical_features = []  # none for this dataset

    return X, y, numeric_features, categorical_features


def build_preprocess(numeric_features: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
        ],
        remainder="drop",
    )


def train_and_select_best(X, y, numeric_features):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
    )

    preprocess = build_preprocess(numeric_features)

    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
        ),
    }

    results = {}
    best_recall = -1.0
    best_name = None
    best_pipeline = None

    # Sample weights (balanced) for training set only
    train_sample_weight = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    for name, model in candidates.items():
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", model),
            ]
        )

        # Apply sample_weight to the model step
        # Works for many sklearn estimators including MLPClassifier and RandomForestClassifier
        pipe.fit(
            X_train,
            y_train,
            model__sample_weight=train_sample_weight,
        )

        y_pred = pipe.predict(X_test)

        # Primary metric: recall for class 1
        recall_1 = recall_score(y_test, y_pred, pos_label=1)

        # Secondary metric: ROC-AUC (if probabilities available)
        auc = None
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)

        results[name] = {
            "recall_class_1": float(recall_1),
            "roc_auc": None if auc is None else float(auc),
            "report": classification_report(y_test, y_pred, output_dict=True),
        }

        if recall_1 > best_recall:
            best_recall = recall_1
            best_name = name
            best_pipeline = pipe

    return best_name, best_pipeline, results


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    target_col = "target"
    X, y, numeric_features, _ = split_features(df, target_col)

    best_name, best_pipe, results = train_and_select_best(X, y, numeric_features)

    if best_pipe is None or best_name is None:
        raise RuntimeError("No best model selected. Check training and metrics.")

    model_path = ARTIFACT_DIR / "heart_disease_pipeline.joblib"
    joblib.dump(best_pipe, model_path)

    meta = {
        "best_model": best_name,
        "selection_metric": "recall_class_1",
        "best_recall_class_1": results[best_name]["recall_class_1"],
        "best_roc_auc": results[best_name]["roc_auc"],
        "features": numeric_features,
        "metrics": results,
    }

    (ARTIFACT_DIR / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print("Best model:", best_name)
    print("Best recall (class 1):", results[best_name]["recall_class_1"])
    print("Best ROC-AUC:", results[best_name]["roc_auc"])
    print("Saved pipeline to:", model_path)
    print("Saved metadata to:", ARTIFACT_DIR / "metadata.json")


if __name__ == "__main__":
    main()
