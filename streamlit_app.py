
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

st.set_page_config(
    page_title="Genetic Disorder Risk Classification",
    page_icon="🧬",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "genetic_disorder_model.pkl"

st.title("🧬 Genetic Disorder Risk Classification System")
st.write("Logistic Regression | Patient Attribute Analysis")

st.warning(
    "Educational and research use only. "
    "This application does not provide a medical diagnosis."
)


# Find the actual trained Pipeline, including inside a dictionary
def find_pipeline(obj):
    if isinstance(obj, Pipeline):
        return obj

    if isinstance(obj, dict):
        for value in obj.values():
            result = find_pipeline(value)
            if result is not None:
                return result

    return None


@st.cache_resource
def load_pipeline():
    saved_object = joblib.load(MODEL_PATH)
    pipeline = find_pipeline(saved_object)

    if pipeline is None:
        raise ValueError(
            "No trained sklearn Pipeline found in model file."
        )

    return pipeline


try:
    model = load_pipeline()
    preprocessor = model.named_steps["preprocessor"]

    model_features = list(preprocessor.feature_names_in_)

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# Numeric columns confirmed from your Colab output
numeric_cols = [
    "Patient Age",
    "Blood cell count (mcL)",
    "Mother's age",
    "Father's age",
    "Test 1",
    "Test 2",
    "Test 3",
    "Test 4",
    "Test 5",
    "No. of previous abortion",
    "White Blood cell count (thousand per microliter)",
    "Symptom 1",
    "Symptom 2",
    "Symptom 3",
    "Symptom 4",
    "Symptom 5",
]

# Only use numeric columns actually present in this model
numeric_cols = [
    col for col in numeric_cols if col in model_features
]

# Every remaining model feature is categorical
categorical_cols = [
    col for col in model_features if col not in numeric_cols
]

st.sidebar.success("Model loaded")
page = st.sidebar.radio(
    "Navigation",
    ["Patient Prediction", "Model Information"]
)


if page == "Patient Prediction":
    st.header("Enter Patient Attributes")
    st.caption("Enter values matching the original training dataset.")

    with st.form("patient_form"):
        patient_data = {}
        col1, col2 = st.columns(2)

        for i, feature in enumerate(model_features):
            container = col1 if i % 2 == 0 else col2

            with container:
                if feature in numeric_cols:
                    patient_data[feature] = st.number_input(
                        feature,
                        value=0.0,
                        key=f"num_{i}"
                    )
                else:
                    patient_data[feature] = st.text_input(
                        feature,
                        value="",
                        key=f"cat_{i}"
                    )

        submitted = st.form_submit_button(
            "Predict Genetic Disorder",
            use_container_width=True
        )

    if submitted:
        patient_df = pd.DataFrame(
            [patient_data],
            columns=model_features
        )

        for feature in numeric_cols:
            patient_df[feature] = pd.to_numeric(
                patient_df[feature],
                errors="coerce"
            )

        try:
            prediction = model.predict(patient_df)[0]

            st.success(f"Predicted Disorder Category: {prediction}")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(patient_df)[0]
                classifier = model.named_steps.get("classifier")

                if classifier is not None and hasattr(classifier, "classes_"):
                    classes = classifier.classes_
                else:
                    classes = list(range(len(probabilities)))

                probability_df = pd.DataFrame({
                    "Category": classes,
                    "Probability": probabilities
                }).sort_values("Probability", ascending=False)

                st.metric(
                    "Highest Model Probability",
                    f"{max(probabilities):.2%}"
                )

                st.bar_chart(
                    probability_df.set_index("Category")["Probability"]
                )

                st.dataframe(
                    probability_df,
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Prediction error: {e}")


else:
    st.header("Model Information")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Preprocessing:** Imputation, scaling and encoding")
    st.write("**Number of input features:**", len(model_features))
    st.write("**Pipeline steps:**", list(model.named_steps.keys()))

    st.dataframe(
        pd.DataFrame({
            "Feature": model_features,
            "Type": [
                "Numeric" if col in numeric_cols else "Categorical"
                for col in model_features
            ]
        }),
        use_container_width=True
    )

    if "Disorder Subclass" in model_features:
        st.warning(
            "'Disorder Subclass' is included in the model input. "
            "If this is the target being predicted, the training "
            "setup must be corrected to avoid target leakage."
        )
