from pathlib import Path

import pandas as pd


RAW_PATH = Path("data/raw/ml-1m/ratings.dat")
OUTPUT_DIR = Path("data/ml-1m")
OUTPUT_PATH = OUTPUT_DIR / "ml-1m.inter"


def main():
    df = pd.read_csv(
        RAW_PATH,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    df = df[["user_id", "item_id", "timestamp"]].sort_values(
        ["user_id", "timestamp"]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False,
        header=["user_id:token", "item_id:token", "timestamp:float"],
    )
    print(f"Saved {len(df):,} interactions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()