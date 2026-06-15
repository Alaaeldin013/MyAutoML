import pandas as pd 

def basic_eda(df):

    eda_results = {}

    # Shape
    eda_results["shape"] = df.shape

    # Data types
    eda_results["dtypes"] = df.dtypes

    # Missing values
    eda_results["missing"] = df.isnull().sum()

    # Statistics
    eda_results["describe"] = df.describe(include="all")

    return eda_results

def visualization(df):
    pass