import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

FEATURES = ["life_expectancy", "mean_years_schooling", "expected_years_schooling", "gni_per_capita"]


def category_from_score(score):
    if score >= 0.800:
        return "Very High"
    if score >= 0.700:
        return "High"
    if score >= 0.550:
        return "Medium"
    return "Low"


def train(dataset_path, model_out_path, metrics_out_path=None):
    df = pd.read_csv(dataset_path)

    X = df[FEATURES].values
    y = df["hdi_score"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)

    pred_cat = [category_from_score(p) for p in preds]
    true_cat = [category_from_score(t) for t in y_test]
    accuracy = float(np.mean([p == t for p, t in zip(pred_cat, true_cat)]))

    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, model_out_path)

    metrics = {
        "algorithm_used": "RandomForestRegressor",
        "accuracy_score": round(accuracy, 4),
        "r2_score": round(float(r2), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    if metrics_out_path:
        with open(metrics_out_path, "w") as f:
            json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(__file__))
    dataset_path = os.path.join(base, "data", "hdi_dataset.csv")
    model_out_path = os.path.join(base, "ml", "hdi_model.pkl")
    metrics_out_path = os.path.join(base, "ml", "metrics.json")

    m = train(dataset_path, model_out_path, metrics_out_path)
    print(json.dumps(m, indent=2))
