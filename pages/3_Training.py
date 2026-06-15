import pandas as pd
import streamlit as st
from src.data.split import split
from src.data.EDA import basic_eda
from src.data.preprocessing import automl_preprocessor
from src.models.models import get_models, train_and_evaluate
from src.utils.utils import detect_task
from config.config import config



st.title("Models Training")

df = st.session_state.get("df")

if df is None:
    st.warning("⚠️ Upload data first")
else:
 
    target = st.selectbox("Select target column: ", df.columns)
  
    y = df[target]

    X = df.drop(target, axis= 1)

    detected_task = detect_task(y)

    st.write(f"Detected Task: **{detected_task.upper()}**")

    task = st.radio(
        "Is this correct?",
        ["Yes", "No"]
    )

    if task == "Yes":
        final_task = detected_task
    if task == "No":
        final_task = st.selectbox(
            "Select the correct task:",
            ["classification", "regression"]
        )


    theprocessor = automl_preprocessor(X)

    

    models_dict = get_models(final_task)

    selected_model_names = st.multiselect(
        "Select models to train:",
        list(models_dict.keys()),
        default=list(models_dict.keys())  # all selected by default
    )

    selected_models = {
        name: models_dict[name]
        for name in selected_model_names
    }

    st.divider()

    if st.button("Train Model"):  
        theprocessor.fit(X, y)
        X_processed = theprocessor.transform(X)

        st.session_state["theprocessor"] = theprocessor

        X_train, X_val, X_test, y_train, y_val, y_test = split(X_processed, y, test_size= config.TEST_SIZE, val_size= config.VAL_SIZE, random_state= config.RANDOM_STATE)

      
        results = {}
        trained_models = {}
        
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        total_models = len(selected_models)
        i = 0
        for name, model in selected_models.items():

            progress = (i + 1) / total_models
            status_text.text(f" Training:  {name}  ({i+1}/{total_models})")
            i = i + 1

            metrics, trained_model = train_and_evaluate(task= final_task, model= model, X_train= X_train, y_train= y_train, X_val= X_val, y_val= y_val)
            
            results[name] = metrics
            trained_models[name] = trained_model

            progress_bar.progress(progress)
            
        status_text.text("All models trained successfully")

        st.session_state["trained_models"] = trained_models

        
        st.divider()
        st.write("###Results")

        for model, metrics in results.items():
            st.write(model)
            st.dataframe(metrics)

        

            
st.divider()


if "trained_models" not in st.session_state:


    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("Back to EDA"):
            st.switch_page("pages/2_EDA.py")

    with col2:
        st.button("Next to Prediction", disabled= True)
            
else:

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("Back to EDA"):
            st.switch_page("pages/2_EDA.py")

    with col2:
        if st.button("Next to Prediction"):
            st.switch_page("pages/4_Prediction.py")



                        

     