
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

st.title("🧬 Genetic Disorder Risk Classification System")
st.write("Logistic Regression | Patient Attribute Analysis")

st.warning(
    "Educational and research use only. "
    "This application does not provide a medical diagnosis."
)

# --------------------------------------------------
# LOAD MODEL FILES
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_columns():
    return joblib.load(FEATURE_PATH)

try:
    model = load_model()
    feature_columns = load_columns()

    # Convert feature names into a regular Python list
    if hasattr(feature_columns, "tolist"):
        feature_columns = feature_columns.tolist()
    else:
        feature_columns = list(feature_columns)

except Exception as e:
    st.error(f"Model files could not be loaded: {e}")
    st.info(
        "Check that genetic_disorder_model.pkl and "
        "feature_columns.pkl exist in the repository root."
    )
    st.stop()

# --------------------------------------------------
# FIND PREPROCESSOR
# --------------------------------------------------

def find_preprocessor(obj):
    if isinstance(obj, ColumnTransformer):
        return obj

    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            result = find_preprocessor(step)
            if result is not None:
                return result

    return None


def get_input_columns(model):
    numeric_cols = []
    categorical_cols = []

    preprocessor = find_preprocessor(model)

    if preprocessor is None:
        return None, None

    for name, transformer, columns in preprocessor.transformers_:
        if isinstance(columns, str):
            columns = [columns]
        elif hasattr(columns, "tolist"):
            columns = columns.tolist()
        else:
            columns = list(columns)

        if transformer == "drop":
            continue

        if transformer == "passthrough":
            categorical_cols.extend(columns)
            continue

        steps = (
            list(transformer.named_steps.values())
            if isinstance(transformer, Pipeline)
            else [transformer]
        )

        if any(isinstance(s, OneHotEncoder) for s in steps):
            categorical_cols.extend(columns)

        elif any(isinstance(s, StandardScaler) for s in steps):
            numeric_cols.extend(columns)

        else:
            # Do not guess unknown column types.
            return None, None

    return numeric_cols, categorical_cols


numeric_cols, categorical_cols = get_input_columns(model)

# --------------------------------------------------
# SIDEBAR NAVIGATION
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
        "Enter values matching the original training dataset columns."
    )

    if numeric_cols is None:
        st.error(
            "The model's preprocessing structure could not be identified. "
            "Please check the training Pipeline before entering predictions."
        )
        st.stop()

    missing_features = [
        col for col in feature_columns
        if col not in numeric_cols and col not in categorical_cols
    ]

    if missing_features:
        st.error(
            "Some input features do not match the model's preprocessing "
            f"configuration: {missing_features}"
        )
        st.stop()

    with st.form("patient_form"):

        patient_data = {}

        col1, col2 = st.columns(2)

        for i, feature in enumerate(feature_columns):

            container = col1 if i % 2 == 0 else col2

            with container:

                if feature in numeric_cols:
                    patient_data[feature] = st.number_input(
                        label=str(feature),
                        value=0.0,
                        key=f"num_{feature}"
                    )

                elif feature in categorical_cols:
                    patient_data[feature] = st.text_input(
                        label=str(feature),
                        value="",
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

            st.success(
                f"Predicted Disorder Category: {prediction}"
            )

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(patient_df)[0]

                classes = getattr(
                    model,
                    "classes_",
                    list(range(len(probabilities)))
                )

                prob_df = pd.DataFrame({
                    "Disorder Category": classes,
                    "Probability": probabilities
                }).sort_values(
                    "Probability",
                    ascending=False
                )

                st.metric(
                    "Highest Model Probability",
                    f"{probabilities.max():.2%}"
                )

                st.subheader("Prediction Probability by Category")

                st.bar_chart(
                    prob_df.set_index("Disorder Category")["Probability"]
                )

                with st.expander("View probability details"):
                    st.dataframe(
                        prob_df,
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
    st.write(
        "**Preprocessing:** Uses the preprocessing steps "
        "saved inside the trained model, when available."
    )
    st.write("**Prediction target:** Genetic Disorder category")

    st.subheader("Input Features")

    st.dataframe(
        pd.DataFrame({"Feature": feature_columns}),
        use_container_width=True
    )

    st.subheader("Model Performance")

    st.info(
        "Display measured test accuracy, precision, recall, F1-score, "
        "and confusion matrix here after loading the actual evaluation "
        "results from training."
    )

    if hasattr(model, "classes_"):
        st.subheader("Supported Prediction Categories")
        st.write([str(c) for c in model.classes_])
