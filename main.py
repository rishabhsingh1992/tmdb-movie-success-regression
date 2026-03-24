import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import joblib

df = pd.read_csv("../../Datasets/tmdb_5000_movies.csv")

df.dropna(inplace=True)

df["profit"] = df["revenue"] - df["budget"]

X = df[
    [
        "budget",
        "popularity",
        "runtime",
    ]
]

y = df["profit"]

(X_train, X_test, y_train, y_test) = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# print(f"R2 Score: {r2_score(y_test, y_pred)}")
# print(f"RMSE: ", rmse)

joblib.dump(model, "models/movie_success_regression_model.pkl")
joblib.dump(scaler, "models/movie_success_regression_scaler.pkl")
