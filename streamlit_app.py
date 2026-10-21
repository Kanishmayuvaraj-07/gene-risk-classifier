import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="Genetic Disorder Risk Classification",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Genetic Disorder Risk Classification System")
st.write("Logistic Regression | Patient Attribute Analysis")
st.warning(
    "Educational and research use only. This application does not "
    "provide a medical diagnosis."
)

@st.cache_resource
def load_model():
    return joblib.load("genetic_disorder_model.pkl")

@st.cache_data
def load_columns():
    return joblib.load("feature_columns.pkl")

try:
    model = load_model()
    feature_columns = load_columns()
except Exception as e:
    st.error(f"Model files could not be loaded: {e}")
    st.stop()

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Patient Prediction", "Model Information"]
)

if page == "Patient Prediction":
    st.header("Enter Patient Attributes")
    st.caption("Enter the values matching the training dataset columns.")

    with st.form("patient_form"):
        patient_data = {}

        numeric_cols = model.named_steps["preprocessor"].transformers_[0][2]
        categorical_cols = model.named_steps["preprocessor"].transformers_[1][2]

        col1, col2 = st.columns(2)

        for i, feature in enumerate(feature_columns):
            container = col1 if i % 2 == 0 else col2

            with container:
                if feature in numeric_cols:
                    patient_data[feature] = st.number_input(
                        feature,
                        value=0.0,
                        key=f"num_{feature}"
                    )
                else:
                    patient_data[feature] = st.text_input(
                        feature,
                        key=f"cat_{feature}"
                    )

        submitted = st.form_submit_button(
            "Predict Genetic Disorder",
            use_container_width=True
        )

    if submitted:
        patient_df = pd.DataFrame(
            [patient_data],
            columns=feature_columns
        )

        try:
            prediction = model.predict(patient_df)[0]
            st.success(f"Predicted Disorder Category: **{prediction}**")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(patient_df)[0]
                confidence = max(probabilities)

                st.metric(
                    "Highest Model Probability",
                    f"{confidence:.2%}"
                )

                prob_df = pd.DataFrame({
                    "Disorder Category": model.classes_,
                    "Probability": probabilities
                }).sort_values("Probability", ascending=False)

                st.subheader("Prediction Probability by Category")
                st.bar_chart(
                    prob_df.set_index("Disorder Category")
                )

        except Exception as e:
            st.error(f"Prediction error: {e}")

elif page == "Model Information":
    st.header("Model Information")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Preprocessing:** Missing-value imputation, scaling, and one-hot encoding.")
    st.write("**Prediction target:** Genetic Disorder category")
    st.write("**Input features:**")
    st.dataframe(pd.DataFrame({"Feature": feature_columns}), use_container_width=True)

    st.subheader("Model Performance")
    st.info(
        "The model's actual test accuracy, precision, recall, F1-score, "
        "and confusion matrix should be saved during training and loaded "
        "here to display the measured results."
    )
