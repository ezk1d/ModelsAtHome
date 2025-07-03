# Author: ezk1d

# ModelsAtHome: Linear Regression

This project is part of the *ModelsAtHome* series, where I learn and reimplement machine learning models from scratch.  
This notebook and code demonstrate a custom implementation of **Linear Regression**, with support for:

- Ordinary Least Squares
- Ridge Regression (L2)
- Lasso Regression (L1)

# ModelsAtHome: Logistic Regression

This project is part of the ModelsAtHome series, where I learn and reimplement machine learning models from scratch for deeper understanding.

Here, I implement **Logistic Regression** using:
- The **sigmoid function** to map predictions to probabilities
- **Gradient descent** for optimization
- Support for **mini-batch training** to reduce training time
- Optional **early stopping** when the loss plateaus

This model is evaluated on the real-world **Breast Cancer Wisconsin** dataset from `sklearn.datasets`.

## ✅ Features

- Vectorized gradient descent
- Support for arbitrary batch sizes (mini-batch, full-batch, or SGD)
- Early stopping with `patience` and `min_delta`
- Intercept term toggle (`fit_intercept`)
- Loss tracking per update
- Evaluation metrics: Accuracy, Precision, Recall, F1 Score
- Model saving and loading

## File Structure

- `models/`: Core implementation
- `notebooks/`: Jupyter notebook to test and explore
- `tests/`: Unit tests

## How to Use

```bash
pip install -r requirements.txt
