from pathlib import Path

from src.data import load_data
from src.features import get_features
from src.model import train_and_evaluate, save_model


DATASET_PATH = Path("../../Datasets/tmdb_5000_movies.csv")
MODEL_PATH = Path("models/movie_success_pipeline.pkl")


df = load_data(DATASET_PATH)
X, y = get_features(df)
pipeline = train_and_evaluate(X, y)
save_model(pipeline, MODEL_PATH)
