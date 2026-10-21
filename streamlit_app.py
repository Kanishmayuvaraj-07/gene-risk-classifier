
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Genetic Disorder Risk Classification",
    page_icon="🧬",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "genetic_disorder_model.pkl"
FEATURE_PATH = BASE_DIR / "feature_columns.pkl"

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🧬 Genetic Disorder Risk Classification System")
st.write("Logistic Regression | Patient Attribute Analysis")

st.warning(
    "Educational and research use only. "
    "This application does not provide a medical diagnosis."
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_feature_columns():
    return joblib.load(FEATURE_PATH)

try:
    model = load_model()
    feature_columns = load_feature_columns()

    if hasattr(feature_columns, "tolist"):
        feature_columns = feature_columns.tolist()
    else:
        feature_columns = list(feature_columns)

except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.info(
        "Ensure genetic_disorder_model.pkl and "
        "feature_columns.pkl are in the same folder as streamlit_app.py."
    )
    st.stop()

# --------------------------------------------------
# GET PREPROCESSOR COLUMNS
# --------------------------------------------------

try:
    preprocessor = model.named_steps["preprocessor"]

    numeric_cols = []
    categorical_cols = []

    for name, transformer, columns in preprocessor.transformers_:

        if name == "remainder":
            continue

        if isinstance(columns, str):
            columns = [columns]
        elif hasattr(columns, "tolist"):
            columns = columns.tolist()
        else:
            columns = list(columns)

        if name == "num":
            numeric_cols.extend(columns)

        elif name == "cat":
            categorical_cols.extend(columns)

    if not numeric_cols and not categorical_cols:
        st.error("Could not identify numeric and categorical input columns.")
        st.stop()

except Exception as e:
    st.error(f"Could not read preprocessing columns: {e}")
    st.stop()

# --------------------------------------------------
# VALIDATE FEATURES
# --------------------------------------------------

unknown_features = [
    col for col in feature_columns
    if col not in numeric_cols and col not in categorical_cols
]

if unknown_features:
    st.error(
        "Some features in feature_columns.pkl do not match the model's "
        f"preprocessor: {unknown_features}"
    )
    st.stop()

# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    ["Patient Prediction", "Model Information"]
)

# --------------------------------------------------
# PATIENT PREDICTION
# --------------------------------------------------

if page == "Patient Prediction":

    st.header("Enter Patient Attributes")
    st.caption(
        "Enter values matching the columns used during model training."
    )

    with st.form("patient_form"):

        patient_data = {}

        col1, col2 = st.columns(2)

        for i, feature in enumerate(feature_columns):

            container = col1 if i % 2 == 0 else col2

            with container:

                if feature in numeric_cols:
                    patient_data[feature] = st.number_input(
                        str(feature),
                        value=0.0,
                        key=f"numeric_{feature}"
                    )

                elif feature in categorical_cols:
                    patient_data[feature] = st.text_input(
                        str(feature),
                        value="",
                        key=f"categorical_{feature}"
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

        # Ensure numeric values are numeric
        for feature in numeric_cols:
            if feature in patient_df.columns:
                patient_df[feature] = pd.to_numeric(
                    patient_df[feature],
                    errors="coerce"
                )

        try:
            prediction = model.predict(patient_df)[0]

            st.success(
                f"Predicted Disorder Category: {prediction}"
            )

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(patient_df)[0]

                classifier = model.named_steps.get(
                    "classifier",
                    model
                )

                classes = getattr(
                    classifier,
                    "classes_",
                    range(len(probabilities))
                )

                probability_df = pd.DataFrame({
                    "Disorder Category": classes,
                    "Probability": probabilities
                }).sort_values(
                    "Probability",
                    ascending=False
                )

                st.metric(
                    "Highest Model Probability",
                    f"{max(probabilities):.2%}"
                )

                st.subheader("Prediction Probability by Category")

                st.bar_chart(
                    probability_df.set_index("Disorder Category")[
                        "Probability"
                    ]
                )

                with st.expander("View probability details"):
                    st.dataframe(
                        probability_df,
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"Prediction error: {e}")

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

elif page == "Model Information":

    st.header("Model Information")

    st.write("**Algorithm:** Logistic Regression")
    st.write("**Preprocessing:** Numeric and categorical transformers")
    st.write("**Prediction target:** Genetic Disorder category")

    st.subheader("Input Features")

    st.dataframe(
        pd.DataFrame({"Feature": feature_columns}),
        use_container_width=True
    )

    st.subheader("Model Structure")

    st.write("Pipeline steps:")

    if hasattr(model, "named_steps"):
        st.write(list(model.named_steps.keys()))

    st.subheader("Model Performance")

    st.info(
        "Actual test accuracy, precision, recall, F1-score, and confusion "
        "matrix must be loaded from the model's evaluation results. "
        "No performance values are assumed here."
    )
