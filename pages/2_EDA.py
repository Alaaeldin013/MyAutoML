import pandas as pd
import streamlit as st
from src.data.EDA import basic_eda
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Exploratory Data Analysis")

df = st.session_state.get("df")

if df is None:
    st.warning("⚠️ Please upload data first")
else:
    st.write("### Data Preview")
    st.dataframe(df.head())

    eda = basic_eda(df)

    st.write("### Shape")
    st.subheader(f"The dataset has ({eda['shape'][0]}) rows and ({eda['shape'][1]}) columns.")


    col1, col2 = st.columns(2)

    with col1:
        st.write("### Data Types")
        st.write(eda["dtypes"])

    with col2:
        st.write("### Missing Values")
        st.write(eda["missing"])

    
    st.write("### Statistics")
    st.write(eda["describe"])

    st.divider()

    # Numeric distributions
    st.write("### Distributions")
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in num_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    st.divider()
    # Correlation
    st.write("### Correlation Heatmap")
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig)

    st.divider()

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("Back to Upload"):
            st.switch_page("pages/1_Upload.py")

    with col2:
        if st.button("Next to Training"):
            st.switch_page("pages/3_Training.py")