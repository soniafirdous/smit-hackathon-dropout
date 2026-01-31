import pandas as pd
import streamlit as st
import joblib

import streamlit as st
import pandas as pd
import joblib

# Load model
@st.cache_resource
def load_model():
    return joblib.load("xgboost_model.pkl")

xgb_pipeline = load_model()

# Risk label
def risk_label(score):
    if score < 0.4:
        return "Low"
    elif score < 0.7:
        return "Medium"
    else:
        return "High"

# Prediction
def predict_dropout(model_pipeline, X_input):
    X = X_input.copy()
    try:
        probs = model_pipeline.predict_proba(X)[:, 1]
    except ValueError as e:
        st.error(f"Prediction failed: {e}")
        return pd.DataFrame()

    preds = (probs >= 0.5).astype(int)
    labels = [risk_label(p) for p in probs]
    student_ids = X.get("student_id", X.index)

    return pd.DataFrame({
        "student_id": student_ids,
        "risk_score": probs,
        "risk_label": labels,
        "predicted_dropout": preds
    })

# Streamlit UI
st.title("📊 Student Dropout Predictor")
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    if st.button("Predict Dropout Risk"):
        results = predict_dropout(xgb_pipeline, df)
        if not results.empty:
            st.subheader("Predictions")
            st.dataframe(results)
