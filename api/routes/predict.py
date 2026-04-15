import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI()


class Movie(BaseModel):
    budget: float
    popularity: float
    runtime: float
    original_language: str
    genres: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    production_companies: list[str] = Field(default_factory=list)
    production_countries: list[str] = Field(default_factory=list)
    release_date: str


model = joblib.load("models/movie_success_pipeline.pkl")


@app.post("/predict")
def predict_movie_success(movie: Movie):
    payload = movie.model_dump()
    payload["genres"] = " ".join(genre.strip().lower() for genre in payload["genres"] if genre.strip())
    payload["keywords"] = " ".join(keyword.strip().lower() for keyword in payload["keywords"] if keyword.strip())
    payload["production_companies"] = " ".join(
        company.strip().lower() for company in payload["production_companies"] if company.strip()
    )
    payload["production_countries"] = " ".join(
        country.strip().lower() for country in payload["production_countries"] if country.strip()
    )

    input_df = pd.DataFrame([payload])
    prediction = model.predict(input_df)

    return {"is_successful": bool(prediction[0] > 0)}
