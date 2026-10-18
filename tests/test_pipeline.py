import json
import pandas as pd
import pytest
from churn_ml.data import generate, validate
from churn_ml.model import make_model, run


def test_deterministic():
    pd.testing.assert_frame_equal(generate(120, 42), generate(120, 42))

def test_different_seed():
    assert not generate(120, 42).equals(generate(120, 43))

def test_schema():
    validate(generate(120))

def test_small_sample_rejected():
    with pytest.raises(ValueError):
        generate(99)

def test_duplicate_rejected():
    df = generate(120)
    df.loc[1, "customer_id"] = df.loc[0, "customer_id"]
    with pytest.raises(ValueError, match="Duplicated"):
        validate(df)

def test_invalid_target_rejected():
    df = generate(120)
    df.loc[0, "churn"] = 5
    with pytest.raises(ValueError, match="Target"):
        validate(df)

def test_models():
    assert make_model("logistic").named_steps["classifier"].__class__.__name__ == "LogisticRegression"

def test_end_to_end(tmp_path):
    report = run(generate(260), tmp_path)
    assert 0 <= report["test"]["roc_auc"] <= 1
    assert sum(report["test"]["confusion_matrix_tn_fp_fn_tp"]) == 52
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "test_predictions.csv").is_file()
    assert (tmp_path / "model.joblib").is_file()
    assert json.loads((tmp_path / "metrics.json").read_text())["dataset"] == "synthetic, fictional"
