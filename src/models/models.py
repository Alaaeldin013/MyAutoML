
import numpy as np
from sklearn.linear_model import (
    LinearRegression,     # Baseline regression
    Ridge,                # L2-regularized regression
    Lasso,                # L1-regularized regression
    ElasticNet            # Combination of L1 and L2 regularization
)
from sklearn.ensemble import (
    RandomForestRegressor,       # Ensemble of decision trees
    GradientBoostingRegressor,   # Boosted trees for regression
    VotingRegressor              # Combine multiple regressors
)
from sklearn.svm import SVR  
from sklearn.metrics import (
    mean_squared_error,  # Regression metric
    r2_score,            # Regression metric
    mean_absolute_error  # Regression metric
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from datetime import datetime
import time  





def get_models(task):
    if task == "classification":
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "AdaBoosting": AdaBoostClassifier(),
            "SVM": SVC()
        }
    else:
        return {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(),
            "ElasticNet": ElasticNet(),
            "GradientBoostingRegressor": GradientBoostingRegressor,
            "SVR": SVR()
        }
    


# ===========================
# 📦 Helper 1 — Basic training and evaluation
# ===========================
def train_and_evaluate(task, model, X_train, y_train, X_val, y_val):
    """Train model and compute basic metrics on train and validation sets."""
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    if task == "regression":
            
        metrics = {
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "val_rmse": np.sqrt(mean_squared_error(y_val, y_val_pred)),
            "train_r2": r2_score(y_train, y_train_pred),
            "val_r2": r2_score(y_val, y_val_pred),
            "train_mae": mean_absolute_error(y_train, y_train_pred),
            "val_mae": mean_absolute_error(y_val, y_val_pred),
            "training_time": training_time,
            "overfitting_gap": r2_score(y_train, y_train_pred) - r2_score(y_val, y_val_pred) # EARLY DETECTION FOR OVERFITTING
        }

    else:
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'train_precision': precision_score(y_train, y_train_pred),
            'val_precision': precision_score(y_val, y_val_pred),
            'train_recall': recall_score(y_train, y_train_pred),
            'val_recall': recall_score(y_val, y_val_pred),
            'train_f1': f1_score(y_train, y_train_pred),
            'val_f1': f1_score(y_val, y_val_pred),
            'training_time': training_time,
            'overfitting_gap': accuracy_score(y_train, y_train_pred) - accuracy_score(y_val, y_val_pred)
        }

    return metrics, model