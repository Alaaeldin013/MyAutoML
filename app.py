
import streamlit as st

st.set_page_config(page_title="AutoML App", layout="wide")

# 🔥 Optional: hide sidebar
st.markdown("""
<style>
[data-testid="stSidebar"] {display: none;}
</style>
""", unsafe_allow_html=True)


# 🎯 HERO SECTION
st.markdown("""
<h1 style='text-align:center;'>My Auto ML</h1>
<p style='text-align:center; font-size:18px; color:gray;'>
Build, Train, and Use Machine Learning Models in Minutes
</p>
""", unsafe_allow_html=True)

st.markdown("---")


# 🧭 HOW IT WORKS (STEP FLOW)
col1, col2, col3, col4 = st.columns(4)

def step(title, desc, icon):
    st.markdown(f"""
    <div style='padding:20px; border-radius:15px; background-color:#1E1E1E; text-align:center'>
        <h2 style='color:#FF9800;'>{icon}</h2>
        <h4 style='color:#4CAF50;'>{title}</h4>
        <p style='color:gray'>{desc}</p>
    </div>
    """, unsafe_allow_html=True)

with col1:
    step("1. Upload", "Upload your CSV dataset", "📂")

with col2:
    step("2. Explore", "Automatic EDA insights", "📊")

with col3:
    step("3. Train", "Train multiple ML models", "🤖")

with col4:
    step("4. Predict", "Generate predictions", "🔮")


st.markdown("---")


# 🚀 MAIN ACTION (ONLY ONE BUTTON)
st.markdown("""
<div style='text-align:center'>
    <h3>Start your journey</h3>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])

with col2:
    if st.button("Upload Your Dataset", use_container_width=True):
        st.switch_page("pages/1_Upload.py")