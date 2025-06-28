import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from linear_regression.core import MyLinearRegression
import numpy as np

def test_fit_and_predict():
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    model = MyLinearRegression()
    model.fit(X, y, fit_intercept=True)
    preds = model.predict(np.column_stack([np.ones(3), X]))
    print(model.theta)
    assert np.allclose(preds, y, atol=1e-2)

test_fit_and_predict()