"""Leakage-safe model selection, thresholding and evaluation."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from .data import FEATURES, NUMERIC, CATEGORICAL, TARGET, validate


def make_model(kind: str):
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    estimator = LogisticRegression(max_iter=1000, random_state=42) if kind == "logistic" else DummyClassifier(strategy="prior")
    return Pipeline([("preprocess", prep), ("classifier", estimator)])


def run(df: pd.DataFrame, output: Path, seed: int = 42) -> dict:
    """Choose model using training CV; reserve test until model choice is fixed."""
    validate(df)
    output.mkdir(parents=True, exist_ok=True)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = {}
    for name in ("baseline", "logistic"):
        model = make_model(name)
        result = cross_validate(model, X_train, y_train, cv=cv, scoring={"roc_auc": "roc_auc", "average_precision": "average_precision"}, n_jobs=1)
        scores[name] = {"roc_auc_mean": float(np.mean(result["test_roc_auc"])), "roc_auc_std": float(np.std(result["test_roc_auc"])), "average_precision_mean": float(np.mean(result["test_average_precision"]))}
    winner = max(scores, key=lambda k: scores[k]["roc_auc_mean"])
    model = make_model(winner).fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    threshold = 0.5  # predeclared business-neutral demonstration threshold; never chosen on test
    predicted = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_test, predicted, labels=[0, 1]).tolist()
    metrics = {"dataset": "synthetic, fictional", "seed": seed, "train_rows": len(X_train), "test_rows": len(X_test), "test_prevalence": float(y_test.mean()), "selection_metric": "training five-fold CV ROC-AUC", "cv": scores, "selected_model": winner, "threshold": threshold, "test": {"roc_auc": float(roc_auc_score(y_test, probabilities)), "average_precision": float(average_precision_score(y_test, probabilities)), "precision": float(precision_score(y_test, predicted, zero_division=0)), "recall": float(recall_score(y_test, predicted, zero_division=0)), "f1": float(f1_score(y_test, predicted, zero_division=0)), "brier_score": float(brier_score_loss(y_test, probabilities)), "confusion_matrix_tn_fp_fn_tp": [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]}}
    permutation = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=seed, scoring="roc_auc", n_jobs=1)
    importance = sorted(zip(FEATURES, permutation.importances_mean.tolist()), key=lambda x: x[1], reverse=True)
    metrics["permutation_importance_roc_auc"] = [{"feature": key, "importance": round(float(value), 5)} for key, value in importance]
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf8")
    pd.DataFrame({"actual_churn": y_test.to_numpy(), "predicted_probability": probabilities.round(6), "predicted_churn": predicted}).to_csv(output / "test_predictions.csv", index=False)
    joblib.dump(model, output / "model.joblib")
    return metrics
