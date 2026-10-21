
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
# LOAD MODEL FILES
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_features():
    return joblib.load(FEATURE_PATH)

try:
    model = load_model()
    feature_columns = load_features()

    if hasattr(feature_columns, "tolist"):
        feature_columns = feature_columns.tolist()
    else:
        feature_columns = list(feature_columns)

except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

# --------------------------------------------------
# FIND PIPELINE AND PREPROCESSOR
# --------------------------------------------------

def find_pipeline(obj):
    if isinstance(obj, Pipeline):
        return obj

    if isinstance(obj, dict):
        for value in obj.values():
            result = find_pipeline(value)
            if result is not None:
                return result

    return None


def find_column_transformer(obj):
    if isinstance(obj, ColumnTransformer):
        return obj

    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            result = find_column_transformer(step)
            if result is not None:
                return result

    if isinstance(obj, dict):
        for value in obj.values():
            result = find_column_transformer(value)
            if result is not None:
                return result

    return None


pipeline = find_pipeline(model)
preprocessor = find_column_transformer(model)

# --------------------------------------------------
# DETERMINE INPUT FEATURES
# --------------------------------------------------

numeric_cols = []
categorical_cols = []

if preprocessor is not None:

    try:
        for name, transformer, columns in preprocessor.transformers_:

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

        if hasattr(preprocessor, "feature_names_in_"):
            model_features = list(preprocessor.feature_names_in_)
        else:
            model_features = list(dict.fromkeys(
                numeric_cols + categorical_cols
            ))

    except Exception as e:
        st.error(f"Could not inspect preprocessing columns: {e}")
        st.stop()

else:
    st.error(
        "No ColumnTransformer was found in the loaded model. "
        "Check that genetic_disorder_model.pkl is the original "
        "trained Pipeline file."
    )

    st.write("Loaded object type:", str(type(model)))

    if isinstance(model, dict):
        st.write("Available dictionary keys:", list(model.keys()))

    st.stop()

if not model_features:
    st.error("No input features were found in the model.")
    st.stop()

missing_features = [
    col for col in model_features
    if col not in feature_columns
]

if missing_features:
    st.error(
        "feature_columns.pkl is missing model input columns. "
        "Please regenerate it from the training notebook."
    )
    st.write("Missing columns:", missing_features)
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
        "Enter values matching the original training dataset."
    )

    with st.form("patient_form"):

        patient_data = {}
        col1, col2 = st.columns(2)

        for i, feature in enumerate(model_features):

            container = col1 if i % 2 == 0 else col2

            with container:

                if feature in numeric_cols:
                    patient_data[feature] = st.number_input(
                        str(feature),
                        value=0.0,
                        key=f"num_{feature}"
                    )

                elif feature in categorical_cols:
                    patient_data[feature] = st.text_input(
                        str(feature),
                        value="",
                        key=f"cat_{feature}"
                    )

                else:
                    st.error(f"Unknown input feature: {feature}")

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

            st.success(
                f"Predicted Disorder Category: {prediction}"
            )

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(patient_df)[0]

                classifier = None

                if pipeline is not None:
                    classifier = pipeline.named_steps.get("classifier")

                classes = (
                    classifier.classes_
                    if classifier is not None
                    and hasattr(classifier, "classes_")
                    else list(range(len(probabilities)))
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
    st.write("**Preprocessing:** Imputation, scaling and encoding")
    st.write("**Prediction target:** Genetic Disorder category")

    st.subheader("Model Input Features")

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

    st.subheader("Model Structure")

    st.write("Loaded object:", str(type(model)))

    if pipeline is not None:
        st.write("Pipeline steps:", list(pipeline.named_steps.keys()))

    st.subheader("Model Performance")

    st.info(
        "Actual test-set performance metrics must be loaded from "
        "the evaluation results. No accuracy values are assumed."
    )
