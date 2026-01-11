import numpy as np
from sklearn.metrics import r2_score
from models.train import evaluate


def test_evaluate_metrics():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    res = evaluate(y_true, y_pred, "unit-test")

    mae_expected = np.mean(np.abs(y_true - y_pred))
    rmse_expected = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2_expected = r2_score(y_true, y_pred)

    assert abs(res["mae"] - mae_expected) < 1e-8
    assert abs(res["rmse"] - rmse_expected) < 1e-8
    assert abs(res["r2"] - r2_expected) < 1e-8
