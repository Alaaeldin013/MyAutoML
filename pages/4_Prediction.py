
import streamlit as st
import pandas as pd

st.title("Prediction")

processor = st.session_state.get("theprocessor")
models = st.session_state.get("trained_models")

if processor is None or models is None:
    st.warning("⚠️ Train models first")

else:

    # 🔥 Model selection
    model_name = st.selectbox(
        "🤖 Choose a model",
        list(models.keys())
    )


    file = st.file_uploader("Upload new data", type=["csv"])


    if file:

        df_new = pd.read_csv(file)


        model = models[model_name]

        try:
            # ✅ Transform only (NO fit)
            Xtopredict = processor.transform(df_new)

            preds = model.predict(Xtopredict)
            
            st.write("### Predictions")
            st.write(preds)

            # Add predictions
            df_new["Prediction"] = preds

            csv = df_new.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Predictions",
                csv,
                "predictions.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")