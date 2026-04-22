from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data import load_data


DATASET_PATH = Path("../../Datasets/tmdb_5000_movies.csv")
TARGET_COLUMN = "profit"


def main() -> None:
    df = load_data(DATASET_PATH)

    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())

    print(f"\n{TARGET_COLUMN} stats:")
    print(df[TARGET_COLUMN].describe())

    plt.figure(figsize=(8, 5))
    df[TARGET_COLUMN].hist(bins=50)
    plt.title("Profit Distribution")
    plt.xlabel(TARGET_COLUMN)
    plt.ylabel("Count")
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    np.log1p(df[TARGET_COLUMN].clip(lower=0)).hist(bins=50)
    plt.title("Log-Transformed Profit Distribution (log1p)")
    plt.xlabel(f"log1p({TARGET_COLUMN})")
    plt.ylabel("Count")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
