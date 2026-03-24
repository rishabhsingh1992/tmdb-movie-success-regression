import joblib
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class Movie(BaseModel):
    budget: float
    popularity: float
    runtime: float


model = joblib.load("models/movie_success_regression_model.pkl")
scaler = joblib.load("models/movie_success_regression_scaler.pkl")


@app.post("/predict")
def predict_movie_success(movie: Movie):
    input_data = [[movie.budget, movie.popularity, movie.runtime]]

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    return {"is_successful": bool(prediction[0])}
