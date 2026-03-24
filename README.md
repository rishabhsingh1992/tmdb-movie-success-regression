# Movie Success Regression

A machine-learning service that predicts whether a movie will be **profitable** (revenue > budget) given its budget, popularity score, and runtime. The model is a regularised linear regression pipeline exposed via a FastAPI REST API.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Training the Model](#training-the-model)
- [Running the API](#running-the-api)
- [API Reference](#api-reference)
- [Dependencies](#dependencies)

---

## Overview

| Component | Technology |
|-----------|-----------|
| Model | `scikit-learn` — `LinearRegression` + `StandardScaler` |
| API | FastAPI + Uvicorn |
| Serialisation | `joblib` |
| Dataset | [TMDB 5000 Movies](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) |

The model is trained on the TMDB 5000 dataset. A preprocessing pipeline standardises features before training. The trained model and scaler are persisted to `models/` and loaded at API startup.

The `POST /predict` endpoint returns a binary profitability signal derived from the continuous regression output (`profit = revenue − budget`).

---

## Project Structure

```
movie-success-regression/
├── api/
│   └── routes/
│       └── predict.py        # FastAPI router — POST /predict
├── models/                   # Serialised model artefacts (git-ignored)
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
# 1. Clone the repository
git clone <repo-url>
cd movie-success-regression

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Training the Model

Place the TMDB dataset CSV (`tmdb_5000_movies.csv`) in the project root, then run:

```bash
python main.py
```

This will:

1. Load and clean the dataset (drop nulls, engineer `profit` feature).
2. Train a `StandardScaler → LinearRegression` pipeline on `budget`, `popularity`, and `runtime`.
3. Evaluate the model (MSE / RMSE / R²) and print results to the console.
4. Serialise the trained model and scaler to `models/`.

---

## Running the API

The model artefacts must exist in `models/` before starting the API (run training first).

```bash
uvicorn api.routes.predict:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs are available at `http://localhost:8000/docs`.

---

## API Reference

### `POST /predict`

Predict whether a movie will be profitable.

**Request body**

```json
{
  "budget":     100000000,
  "popularity": 85.5,
  "runtime":    132
}
```

| Field | Type | Description |
|-------|------|-------------|
| `budget` | `float` | Production budget in USD |
| `popularity` | `float` | TMDB popularity score |
| `runtime` | `float` | Film duration in minutes |

**Response**

```json
{
  "is_successful": true
}
```

`is_successful` is `true` when the predicted profit (revenue − budget) is positive.

---

## Dependencies

Key packages — see `requirements.txt` for pinned versions.

| Package | Purpose |
|---------|---------|
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `scikit-learn` | ML pipeline |
| `pandas` / `numpy` | Data processing |
| `joblib` | Model serialisation |
| `pydantic` | Request validation |
