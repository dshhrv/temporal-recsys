import json
from pathlib import Path

import numpy as np
import pandas as pd


RAW_PATH = Path("data/raw/ml-1m/ratings.dat")

OUTPUT_DIR = Path("data/tgn_ml1m")
INTERACTIONS_PATH = OUTPUT_DIR / "interactions.csv"
NODE_MAPPING_PATH = OUTPUT_DIR / "node_mapping.csv"
META_PATH = OUTPUT_DIR / "meta.json"

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1


def main():
    ratings = pd.read_csv(
        RAW_PATH,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={
            "user_id": "int64",
            "item_id": "int64",
            "rating": "int64",
            "timestamp": "int64",
        },
        encoding="latin-1",
    )

    ratings["_original_order"] = np.arange(len(ratings))

    ratings = ratings.sort_values(
        ["timestamp", "_original_order"],
        kind="stable",
    ).reset_index(drop=True)

    n_events = len(ratings)

    train_end = int(n_events * TRAIN_RATIO)
    valid_end = int(n_events * (TRAIN_RATIO + VALID_RATIO))

    ratings["split"] = "test"
    ratings.loc[:train_end - 1, "split"] = "train"
    ratings.loc[train_end:valid_end - 1, "split"] = "valid"

    train_item_ids = set(ratings.loc[ratings["split"] == "train", "item_id"].unique())
    cold_item_mask = (ratings["split"] != "train") & ~ratings["item_id"].isin(train_item_ids)
    removed_cold_counts = ratings.loc[cold_item_mask, "split"].value_counts().to_dict()
    ratings = ratings.loc[~cold_item_mask].reset_index(drop=True)

    user_ids = np.sort(ratings["user_id"].unique())
    item_ids = np.sort(ratings["item_id"].unique())

    n_users = len(user_ids)
    n_items = len(item_ids)

    user_index = pd.Index(user_ids)
    item_index = pd.Index(item_ids)

    ratings["src"] = user_index.get_indexer(ratings["user_id"]).astype("int64")
    ratings["dst"] = (n_users + item_index.get_indexer(ratings["item_id"])).astype("int64")
    ratings["event_id"] = np.arange(len(ratings), dtype=np.int64)
    ratings["time_days"] = ((ratings["timestamp"] - ratings["timestamp"].min()) / 86400.0).astype("float32")
    ratings["edge_feat_0"] = (ratings["rating"] / 5.0).astype("float32")

    interactions = ratings[
        [
            "event_id",
            "user_id",
            "item_id",
            "src",
            "dst",
            "timestamp",
            "time_days",
            "rating",
            "edge_feat_0",
            "split",
        ]
    ]

    user_mapping = pd.DataFrame(
        {
            "node_id": np.arange(n_users),
            "node_type": "user",
            "original_id": user_ids,
        }
    )

    item_mapping = pd.DataFrame(
        {
            "node_id": np.arange(n_users, n_users + n_items),
            "node_type": "item",
            "original_id": item_ids,
        }
    )

    node_mapping = pd.concat(
        [user_mapping, item_mapping],
        ignore_index=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    interactions.to_csv(INTERACTIONS_PATH, index=False)
    node_mapping.to_csv(NODE_MAPPING_PATH, index=False)

    metadata = {
        "dataset": "MovieLens-1M",
        "n_interactions": int(n_events),
        "n_interactions_after_cold_item_filter": int(len(ratings)),
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_nodes": int(n_users + n_items),
        "edge_feature_dim": 1,
        "edge_feature": "rating / 5.0",
        "time_unit": "days",
        "split": {
            "method": "global chronological 80/10/10, then remove validation/test cold items",
            "train": int((ratings["split"] == "train").sum()),
            "valid": int((ratings["split"] == "valid").sum()),
            "test": int((ratings["split"] == "test").sum()),
            "raw_train": int(train_end),
            "raw_valid": int(valid_end - train_end),
            "raw_test": int(n_events - valid_end),
            "removed_cold_valid": int(removed_cold_counts.get("valid", 0)),
            "removed_cold_test": int(removed_cold_counts.get("test", 0)),
        },
    }

    META_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
