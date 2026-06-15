from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd


def automl_preprocessor(X):

    # 🔍 Detect column types
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
    datetime_cols = X.select_dtypes(include=['datetime64']).columns.tolist()

    # 🔥 Convert boolean → categorical
    cat_cols += bool_cols

    # 🚀 Detect high-cardinality categorical columns
    high_card_cols = [col for col in cat_cols if X[col].nunique() > 20]
    low_card_cols = [col for col in cat_cols if X[col].nunique() <= 20]

    # ------------------------
    # 🔢 Numerical pipeline
    # ------------------------
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    # ------------------------
    # 🧩 Low-card categorical
    # ------------------------
    low_cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # ------------------------
    # 🔥 High-card categorical (simplified handling)
    # ------------------------
    high_cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # fallback to OneHot (can replace later with target encoding)
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # ------------------------
    # 📅 Datetime handling (basic)
    # ------------------------
    def extract_datetime_features(df):
        df = df.copy()
        for col in df.columns:
            df[col + "_year"] = df[col].dt.year
            df[col + "_month"] = df[col].dt.month
            df[col + "_day"] = df[col].dt.day
        return df.drop(columns=df.columns)

    # ------------------------
    # 🧱 Combine everything
    # ------------------------
    transformers = []

    if num_cols:
        transformers.append(('num', num_pipeline, num_cols))

    if low_card_cols:
        transformers.append(('low_cat', low_cat_pipeline, low_card_cols))

    if high_card_cols:
        transformers.append(('high_cat', high_cat_pipeline, high_card_cols))

    preprocessor = ColumnTransformer(transformers)

    return preprocessor
