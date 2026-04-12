import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. Load data
df = pd.read_csv("../../Datasets/tmdb_5000_movies.csv")

# 2. Basic Cleaning
df["profit"] = df["revenue"] - df["budget"]

df = df.dropna()

X = df[["budget", "popularity", "runtime", "original_language"]]
y = df["profit"]

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Define Preprocessing
NUMERIC_COLS = ["budget", "popularity", "runtime"]
CATEGORICAL_COLS = ["original_language"]

preprocessor = ColumnTransformer(
    [
        ("numeric", StandardScaler(), NUMERIC_COLS),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ],
)

# 5. Pipeline & Training
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ]
)

pipeline.fit(X_train, y_train)

# 6. Evaluate
y_pred = pipeline.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

# 7. Save the WHOLE pipeline
joblib.dump(pipeline, "models/movie_success_pipeline.pkl")
