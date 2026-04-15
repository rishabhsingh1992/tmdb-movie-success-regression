import pandas as pd


def extract_release_date_features(release_dates: pd.DataFrame) -> pd.DataFrame:
    """Extract numerical calendar features from TMDB release date values."""
    if isinstance(release_dates, pd.DataFrame):
        values = release_dates.iloc[:, 0]
    else:
        values = pd.Series(release_dates)

    parsed_dates = pd.to_datetime(values, errors="coerce")

    return pd.DataFrame(
        {
            "release_year": parsed_dates.dt.year.fillna(0).astype(int),
            "release_month": parsed_dates.dt.month.fillna(0).astype(int),
        },
        index=values.index,
    )


def get_features(df: pd.DataFrame):
    """Return feature matrix X and target vector y."""
    X = df[
        [
            "budget",
            "popularity",
            "runtime",
            "original_language",
            "genres",
            "keywords",
            "production_companies",
            "production_countries",
            "release_date",
        ]
    ]
    y = df["profit"]
    return X, y
