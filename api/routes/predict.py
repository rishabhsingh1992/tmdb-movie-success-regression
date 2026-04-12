import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class Movie(BaseModel):
    budget: float
    popularity: float
    runtime: float
    original_language: str


model = joblib.load("models/movie_success_pipeline.pkl")


@app.post("/predict")
def predict_movie_success(movie: Movie):
    input_df = pd.DataFrame([movie.model_dump()])

    prediction = model.predict(input_df)

    return {"is_successful": bool(prediction[0] > 0)}
