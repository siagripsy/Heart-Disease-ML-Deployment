from __future__ import annotations

from pathlib import Path
import pandas as pd


RAW_PATH = Path("data/raw/heart.csv")
OUT_DIR = Path("data/processed")
OUT_PATH = OUT_DIR / "heart_clean.csv"

TARGET_ORIGINAL = "HeartDiseaseorAttack"


def clean_heart_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Clean column names (no semantic changes)
    df.columns = [c.strip() for c in df.columns]

    # 2) Drop duplicated rows
    df = df.drop_duplicates()

    # 3) Check target existence
    if TARGET_ORIGINAL not in df.columns:
        raise ValueError(f"Target column '{TARGET_ORIGINAL}' not found.")

    # 4) Rename target to standard name
    df = df.rename(columns={TARGET_ORIGINAL: "target"})

    # 5) Convert columns to numeric safely (no casting to int)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 6) Drop rows with missing target
    df = df.dropna(subset=["target"])

    # 7) Force target to int (binary classification)
    df["target"] = df["target"].astype(int)

    return df


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    df_clean = clean_heart_data(df)

    df_clean.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Rows:", len(df_clean), "Cols:", df_clean.shape[1])
    print("Dtypes:")
    print(df_clean.dtypes)
    print("\nTarget distribution:")
    print(df_clean["target"].value_counts())


if __name__ == "__main__":
    main()
