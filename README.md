# Movie Success Regression

A machine-learning service that predicts whether a movie will be **profitable** (`revenue > budget`) using budget, popularity, runtime, original language, and genres. The model is a linear regression pipeline exposed via a FastAPI REST API.

---

## Overview

| Component | Technology |
|-----------|-----------|
| Model | `scikit-learn` — `LinearRegression` |
| Feature preprocessing | `StandardScaler` + `OneHotEncoder` + `TfidfVectorizer` |
| API | FastAPI + Uvicorn |
| Serialisation | `joblib` |
| Dataset | [TMDB 5000 Movies](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) |

The model is trained on the TMDB 5000 dataset. The training pipeline now parses the raw TMDB `genres` JSON-like payload and uses the normalized genre names as a text feature.

---

## Project Structure

```text
movie-success-regression/
├── api/
│   └── routes/
│       └── predict.py        # FastAPI app — POST /predict
├── models/                   # Serialized model artefacts (git-ignored)
├── main.py                   # Model training script
├── requirements.txt
├── .gitignore
├── README.md
└── TASKS.md
```

---

## Setup

**Prerequisites:** Python 3.10+

```bash
git clone <repo-url>
cd movie-success-regression
python -m venv .venv
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

---

## Training the Model

Place `tmdb_5000_movies.csv` in `../../Datasets/`, then run:

```bash
python main.py
```

This will:

1. Engineer `profit = revenue - budget`.
2. Parse and normalize `genres` values.
3. Train a preprocessing + regression pipeline on `budget`, `popularity`, `runtime`, `original_language`, and `genres`.
4. Print MAE, RMSE, and R².
5. Save the trained pipeline to `models/movie_success_pipeline.pkl`.

---

## Running the API

```bash
uvicorn api.routes.predict:app --reload --host 0.0.0.0 --port 8000
```

Docs: `http://localhost:8000/docs`

---

## API Reference

### `POST /predict`

**Request body**

```json
{
  "budget": 100000000,
  "popularity": 85.5,
  "runtime": 132,
  "original_language": "en",
  "genres": ["Action", "Adventure", "Science Fiction"]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `budget` | `float` | Production budget in USD |
| `popularity` | `float` | TMDB popularity score |
| `runtime` | `float` | Film duration in minutes |
| `original_language` | `str` | ISO language code |
| `genres` | `list[str]` | List of genre labels |

**Response**

```json
{
  "is_successful": true
}
```

`is_successful` is `true` when predicted profit is positive.
