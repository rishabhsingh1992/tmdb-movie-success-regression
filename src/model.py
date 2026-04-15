from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features import extract_release_date_features


NUMERIC_COLS = ["budget", "popularity", "runtime"]
CATEGORICAL_COLS = ["original_language"]
RELEASE_DATE_COL = ["release_date"]


def build_pipeline() -> Pipeline:
    """Build and return the full sklearn preprocessing + model pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("genres", TfidfVectorizer(), "genres"),
            ("keywords", TfidfVectorizer(), "keywords"),
            ("production_companies", TfidfVectorizer(), "production_companies"),
            ("production_countries", TfidfVectorizer(), "production_countries"),
            (
                "release_date",
                Pipeline(
                    [
                        (
                            "date_features",
                            FunctionTransformer(
                                extract_release_date_features, validate=False
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                RELEASE_DATE_COL,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def train_and_evaluate(X, y) -> Pipeline:
    """Split data, train the pipeline, print evaluation metrics, and return the fitted pipeline."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

    return pipeline


def save_model(pipeline: Pipeline, path: Path):
    """Save the fitted pipeline to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
