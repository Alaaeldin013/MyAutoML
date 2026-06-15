import pandas as pd
import streamlit as st

st.title("📂 Upload Dataset")

file = st.file_uploader("Upload your CSV file", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.session_state["df"] = df

    st.success("✅ File uploaded successfully!")

    st.write("### Preview")
    st.dataframe(df.head(15))

    st.write("### Shape")
    st.write(df.shape)




    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("Back to Home Page"):
            st.switch_page("Home.py")

    with col2:
        if st.button("Next to EDA"):
            st.switch_page("pages/2_EDA.py")