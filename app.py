# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 1. Load Model & Preprocessor
# ===============================
@st.cache_resource
def load_model():
    model = joblib.load('xgboost_model.pkl')
    return model

xgb_pipeline = load_model()

# ===============================
# 2. Prediction Function
# ===============================
def risk_label(score):
    if score < 0.4: return "Low"
    elif score < 0.7: return "Medium"
    else: return "High"

def predict_dropout(model_pipeline, X_input):
    X_input = X_input.copy()
    
    # Derived features
    X_input['engagement_score'] = (
        X_input.get('raisedhands', 0) +
        X_input.get('VisITedResources', 0) +
        X_input.get('AnnouncementsView', 0) +
        X_input.get('Discussion', 0)
    )
    X_input['class_participation'] = X_input.get('raisedhands', 0) + X_input.get('VisITedResources', 0)

    # Map absence
    if X_input['StudentAbsenceDays'].dtype == object:
        X_input['StudentAbsenceDays'] = X_input['StudentAbsenceDays'].map({'Under-7': 0, 'Above-7': 1})

    student_ids = X_input.get('student_id', X_input.index)

    probs = model_pipeline.predict_proba(X_input)[:, 1]
    preds = (probs >= 0.5).astype(int)
    labels = [risk_label(p) for p in probs]

    return pd.DataFrame({
        'student_id': student_ids,
        'risk_score': probs,
        'risk_label': labels,
        'predicted_dropout': preds
    })

# ===============================
# 3. Streamlit UI
# ===============================
st.set_page_config(page_title="Student Dropout Early Warning System", layout="wide")
st.title("📊 Student Dropout Early Warning System")

st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload a CSV file containing student data.
2. Make sure the CSV contains all required columns:
   - gender, NationalITy, PlaceofBirth, StageID, GradeID, SectionID
   - Topic, Semester, Relation, raisedhands, VisITedResources
   - AnnouncementsView, Discussion, ParentAnsweringSurvey
   - ParentschoolSatisfaction, StudentAbsenceDays
3. Click 'Predict Dropout Risk' to see results.
""")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    students_df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(students_df.head())

    if st.button("Predict Dropout Risk"):
        results_df = predict_dropout(xgb_pipeline, students_df)

        st.subheader("Top High-Risk Students")
        top_risk = results_df.sort_values('risk_score', ascending=False).head(20)
        st.dataframe(top_risk)

        st.subheader("All Predictions")
        st.dataframe(results_df)

        # Optional: Risk distribution plot
        st.subheader("Risk Distribution")
        plt.figure(figsize=(8,4))
        sns.countplot(x='risk_label', data=results_df, order=['Low','Medium','High'])
        plt.title("Number of Students by Risk Level")
        st.pyplot(plt)
