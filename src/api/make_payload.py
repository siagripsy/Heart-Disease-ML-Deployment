import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_clean.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    row = df.drop(columns=["target"], errors="ignore").iloc[0].to_dict()

    payload = {"features": row}

    out_path = PROJECT_ROOT / "artifacts" / "sample_payload.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("Saved sample payload to:", out_path)

if __name__ == "__main__":
    main()
