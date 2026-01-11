import joblib
import numpy as np
from sklearn.dummy import DummyRegressor


def test_joblib_save_load(tmp_path):
    model = DummyRegressor(strategy="mean")
    # fit on tiny dataset so model has learned state
    X = [[1.0], [2.0], [3.0]]
    y = [1.0, 2.0, 3.0]
    model.fit(X, y)

    p = tmp_path / "tmp_model.joblib"
    joblib.dump(model, p)
    loaded = joblib.load(p)

    assert type(loaded) is type(model)
    # predictions should match
    assert np.allclose(loaded.predict([[4.0]]), model.predict([[4.0]]))
