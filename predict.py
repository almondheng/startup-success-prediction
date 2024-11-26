import pandas as pd
import cloudpickle
from flask import Flask, request, jsonify


def load_model(model_path="startup_success_model.pkl"):
    try:
        with open(model_path, "rb") as f:
            model = cloudpickle.load(f)
            return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        raise


def load_label_mapping(encoder_path="status_label_encoder.pkl"):
    try:
        with open(encoder_path, "rb") as f:
            le = cloudpickle.load(f)
            label_mapping = dict(enumerate(le.classes_))
            return label_mapping
    except FileNotFoundError:
        print(f"Error: Target label encoder file not found at {encoder_path}")
        return None


def preprocess_input_data(input_data: dict):
    required_fields = [
        "funding_total_usd",
        "funding_rounds",
        "country_code",
        "state_code",
        "region",
        "city",
        "category_list",
        "first_funding_at",
        "last_funding_at",
    ]

    for field in required_fields:
        if field not in input_data:
            raise ValueError(f"Missing required field: {field}")

    df = pd.DataFrame([input_data])

    df.funding_total_usd = df.funding_total_usd.replace("-", -1)
    df.funding_total_usd = pd.to_numeric(df.funding_total_usd, errors="coerce")
    df["is_undisclosed"] = (df.funding_total_usd == -1).astype(int)

    # Convert dates
    df.first_funding_at = pd.to_datetime(df.first_funding_at, errors="coerce")
    df.last_funding_at = pd.to_datetime(df.last_funding_at, errors="coerce")

    # Add funding duration
    df["funding_duration"] = (df.last_funding_at - df.first_funding_at).dt.days

    # Fill missing values
    df.category_list = df.category_list.fillna("Unknown")
    df.state_code = df.state_code.fillna("Unknown")
    df.region = df.region.fillna("Unknown")

    # Split categories
    df["category_list_split"] = df["category_list"].str.split("|")

    # Drop unnecessary columns
    df.drop(
        ["first_funding_at", "last_funding_at", "category_list"], axis=1, inplace=True
    )

    return df


def predict_startup_success(model, preprocessed_data):
    probabilities = model.predict_proba(preprocessed_data)
    predictions = model.predict(preprocessed_data)

    return predictions, probabilities


model = load_model()
label_mapping = load_label_mapping()

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected JSON input:
    {
        "funding_total_usd": 5500000,
        "funding_rounds": 2,
        "country_code": "US",
        "state_code": "CA",
        "region": "Silicon Valley",
        "city": "San Francisco",
        "category_list": "Technology|Software",
        "first_funding_at": "2022-01-01",
        "last_funding_at": "2023-01-01",
    }
    """
    input_data = request.get_json()

    if not input_data:
        return jsonify({"error": "No input data provided"}), 400

    preprocessed_data = preprocess_input_data(input_data)

    predictions, probabilities = predict_startup_success(model, preprocessed_data)

    if label_mapping:
        prediction_labels = [label_mapping[pred] for pred in predictions]

    response = {
        "prediction": prediction_labels[0] if label_mapping else str(predictions[0]),
        "probabilities": {
            label_mapping.get(i, str(i)): float(prob)
            for i, prob in enumerate(probabilities[0])
        },
    }

    return jsonify(response)
