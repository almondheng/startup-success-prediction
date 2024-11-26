import cloudpickle
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import (
    LabelEncoder,
    RobustScaler,
    OneHotEncoder,
    MultiLabelBinarizer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier


class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None):
        self.mlb.fit(X.iloc[:, 0].tolist())
        return self

    def transform(self, X):
        return self.mlb.transform(X.iloc[:, 0].tolist())

    def get_feature_names_out(self, input_features=None):
        return self.mlb.classes_


def load_and_preprocess_data(file_path):
    # Load data
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    df.drop(["permalink", "name", "homepage_url"], axis=1, inplace=True)

    # Process funding total USD
    df.funding_total_usd = df.funding_total_usd.replace("-", -1)
    df.funding_total_usd = pd.to_numeric(df.funding_total_usd, errors="coerce")

    # Convert date columns
    df.first_funding_at = pd.to_datetime(df.first_funding_at, errors="coerce")
    df.last_funding_at = pd.to_datetime(df.last_funding_at, errors="coerce")

    # Fill missing values
    df.category_list = df.category_list.fillna("Unknown")
    df.drop(["founded_at"], axis=1, inplace=True)

    # Drop rows with missing values
    df = df.dropna(subset="first_funding_at")
    df = df.dropna(subset=["country_code", "city"], how="any")

    df.state_code = df.state_code.fillna("Unknown")
    df.region = df.region.fillna("Unknown")

    return df


def feature_engineering(df):
    # Remove outliers
    df_feature = df[df.funding_total_usd < 3000].copy()

    # Add is_undisclosed column
    df_feature["is_undisclosed"] = (df_feature.funding_total_usd == -1).astype(int)

    # Convert date columns to duration
    df_feature["funding_duration"] = (
        df_feature.last_funding_at - df_feature.first_funding_at
    ).dt.days
    df_feature.drop(["first_funding_at", "last_funding_at"], axis=1, inplace=True)

    # Split categories
    df_feature["category_list_split"] = df_feature["category_list"].str.split("|")
    df_feature.drop(["category_list"], axis=1, inplace=True)

    # Encode target variable
    le = LabelEncoder()
    df_feature.status = le.fit_transform(df_feature.status)

    return df_feature, le


def create_preprocessor():
    column_transformer = ColumnTransformer(
        transformers=[
            (
                "num",
                RobustScaler(),
                ["funding_total_usd", "funding_rounds", "funding_duration"],
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
                ["country_code", "state_code", "region", "city"],
            ),
            ("categories", MultiLabelBinarizerTransformer(), ["category_list_split"]),
        ],
        remainder="passthrough",
    )
    return column_transformer


def train_model_with_grid_search(X, y):
    # Create preprocessor
    preprocessor = create_preprocessor()

    # Define XGBoost model
    model = XGBClassifier(
        objective="multi:softmax", num_class=len(np.unique(y)), random_state=42
    )

    # Create pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # Define parameter grid for grid search
    param_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.1, 0.2],
    }

    # Perform Grid Search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )

    # Fit GridSearchCV
    grid_search.fit(X, y)

    return grid_search


def main():
    # File path - replace with your actual file path
    file_path = "big_startup_secsees_dataset.csv"

    # Load and preprocess data
    df = load_and_preprocess_data(file_path)

    # Feature engineering
    df_feature, status_label_encoder = feature_engineering(df)

    # Prepare features and target
    X = df_feature.drop("status", axis=1)
    y = df_feature.status

    # Train model
    grid_search = train_model_with_grid_search(X, y)

    # Print results
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_}")

    with open("startup_success_model.pkl", "wb") as f:
        cloudpickle.dump(grid_search.best_estimator_, f)
        print("Best model saved as startup_success_model.pkl")

    with open("status_label_encoder.pkl", "wb") as f:
        cloudpickle.dump(status_label_encoder, f)
        print("Target label encoder saved as status_label_encoder.pkl")


if __name__ == "__main__":
    main()
