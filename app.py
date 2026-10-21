from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("genetic_disorder_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Genetic Disorder Classification API is running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "JSON input is required"}), 400

        input_df = pd.DataFrame([data])
        input_df = input_df.reindex(columns=feature_columns)

        prediction = model.predict(input_df)[0]

        result = {"predicted_genetic_disorder": str(prediction)}

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            result["prediction_probability"] = float(max(probabilities))

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
