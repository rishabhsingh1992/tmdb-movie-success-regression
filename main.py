import ast
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASET_PATH = Path("../../Datasets/tmdb_5000_movies.csv")
MODEL_PATH = Path("models/movie_success_pipeline.pkl")


def parse_genres(raw_genres: str) -> str:
    """Convert TMDB JSON-like genres payload into a space-separated genre string."""
    if not isinstance(raw_genres, str) or not raw_genres.strip():
        return ""

    payload = raw_genres.strip()
    parsed = None

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(payload)
        except (ValueError, SyntaxError):
            return ""

    if not isinstance(parsed, list):
        return ""

    genre_names = [item.get("name", "") for item in parsed if isinstance(item, dict)]
    genre_names = [name.strip().lower() for name in genre_names if isinstance(name, str) and name.strip()]
    return " ".join(genre_names)


# 1. Load data
raw_df = pd.read_csv(DATASET_PATH)

# 2. Basic cleaning + feature engineering
df = raw_df.copy()
df["profit"] = df["revenue"] - df["budget"]
df["genres"] = df["genres"].apply(parse_genres)

df = df.dropna(subset=["budget", "popularity", "runtime", "original_language", "genres", "profit"])

X = df[["budget", "popularity", "runtime", "original_language", "genres"]]
y = df["profit"]

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Define preprocessing
NUMERIC_COLS = ["budget", "popularity", "runtime"]
CATEGORICAL_COLS = ["original_language"]
TEXT_COL = "genres"

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), NUMERIC_COLS),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ("genres", TfidfVectorizer(), TEXT_COL),
    ]
)

# 5. Pipeline & training
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression()),
])

pipeline.fit(X_train, y_train)

# 6. Evaluate
y_pred = pipeline.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

# 7. Save the whole pipeline
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
