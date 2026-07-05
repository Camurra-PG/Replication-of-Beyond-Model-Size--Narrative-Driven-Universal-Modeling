from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


EVENT_NAME_MAP = {
    "view": "page_visit",
    "addtocart": "add_to_cart",
    "transaction": "product_buy",
}


def load_retailrocket_events(data_dir: Path) -> pl.DataFrame:
    events_path = data_dir / "events.csv"
    part1_path = data_dir / "item_properties_part1.csv"
    part2_path = data_dir / "item_properties_part2.csv"

    if not events_path.exists():
        raise FileNotFoundError(f"Missing events file: {events_path}")

    if not part1_path.exists() or not part2_path.exists():
        raise FileNotFoundError(
            "Missing item_properties_part1.csv or item_properties_part2.csv"
        )

    print("Loading Retailrocket events...")

    events = (
        pl.scan_csv(events_path)
        .select(
            [
                pl.col("visitorid").cast(pl.Int64).alias("client_id"),
                pl.from_epoch(
                    pl.col("timestamp").cast(pl.Int64),
                    time_unit="ms",
                ).alias("timestamp"),
                pl.col("event").replace(EVENT_NAME_MAP).alias("event_type"),
                pl.col("itemid").cast(pl.Int64).alias("sku"),
                pl.col("transactionid")
                .cast(pl.Int64, strict=False)
                .alias("transaction_id"),
            ]
        )
    )

    print("Loading latest category assignments...")

    categories = (
        pl.concat(
            [
                pl.scan_csv(part1_path),
                pl.scan_csv(part2_path),
            ]
        )
        .filter(pl.col("property") == "categoryid")
        .select(
            [
                pl.col("itemid").cast(pl.Int64).alias("sku"),
                pl.from_epoch(
                    pl.col("timestamp").cast(pl.Int64),
                    time_unit="ms",
                ).alias("property_timestamp"),
                pl.col("value")
                .cast(pl.Int64, strict=False)
                .alias("category_id"),
            ]
        )
        .filter(pl.col("category_id").is_not_null())
        .sort(["sku", "property_timestamp"])
        .group_by("sku")
        .agg(pl.col("category_id").last().alias("category_id"))
    )

    events = (
        events.join(categories, on="sku", how="left")
        .collect(engine="streaming")
        .sort(["client_id", "timestamp"])
    )

    return events


def top_ids_and_counts(
    df: pl.DataFrame,
    id_col: str,
    count_name: str,
    top_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    if df.is_empty() or id_col not in df.columns:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    counts = (
        df.filter(pl.col(id_col).is_not_null())
        .group_by(id_col)
        .agg(pl.len().alias(count_name))
        .sort(count_name, descending=True)
        .head(top_n)
    )

    ids = counts[id_col].to_numpy().astype(np.int64)
    popularity = counts[count_name].to_numpy().astype(np.int64)

    return ids, popularity


def build_target_frame(
    clients: np.ndarray,
    history_buyers: set[int],
    target_active_clients: set[int],
    target_buyers: set[int],
    target_purchases: pl.DataFrame,
) -> pl.DataFrame:
    """
    Build one target table.

    Columns:
    - client_id
    - churn:
        1 = historical buyer did not buy again in target window
        0 = historical buyer bought again in target window
        null = no historical purchase, churn not defined
    - active:
        1 = any target-window event
        0 = no target-window event
    - propensity_category:
        list of category_ids purchased by the client in target window
    - propensity_sku:
        list of skus purchased by the client in target window
    - propensity_new_sku:
        list of target-window purchased skus that are globally new
        with respect to historical purchases
    """

    client_df = pl.DataFrame({"client_id": clients.astype(np.int64)})

    if target_purchases.is_empty():
        labels = client_df.with_columns(
            [
                pl.lit(None).cast(pl.Int8).alias("churn"),
                pl.lit(0).cast(pl.Int8).alias("active"),
                pl.lit([]).cast(pl.List(pl.Int64)).alias("propensity_category"),
                pl.lit([]).cast(pl.List(pl.Int64)).alias("propensity_sku"),
                pl.lit([]).cast(pl.List(pl.Int64)).alias("propensity_new_sku"),
            ]
        )
        return labels

    per_client = (
        target_purchases.group_by("client_id")
        .agg(
            [
                pl.col("category_id")
                .drop_nulls()
                .unique()
                .sort()
                .alias("propensity_category"),
                pl.col("sku")
                .drop_nulls()
                .unique()
                .sort()
                .alias("propensity_sku"),
                pl.col("new_sku")
                .drop_nulls()
                .unique()
                .sort()
                .alias("propensity_new_sku"),
            ]
        )
    )

    labels = client_df.join(per_client, on="client_id", how="left")

    labels = labels.with_columns(
        [
            pl.col("propensity_category")
            .fill_null([])
            .cast(pl.List(pl.Int64)),
            pl.col("propensity_sku")
            .fill_null([])
            .cast(pl.List(pl.Int64)),
            pl.col("propensity_new_sku")
            .fill_null([])
            .cast(pl.List(pl.Int64)),
        ]
    )

    labels = labels.with_columns(
        [
            pl.col("client_id")
            .map_elements(
                lambda cid: 1 if int(cid) in target_active_clients else 0,
                return_dtype=pl.Int8,
            )
            .alias("active"),
            pl.col("client_id")
            .map_elements(
                lambda cid: (
                    None
                    if int(cid) not in history_buyers
                    else (0 if int(cid) in target_buyers else 1)
                ),
                return_dtype=pl.Int8,
            )
            .alias("churn"),
        ]
    )

    return labels.select(
        [
            "client_id",
            "churn",
            "active",
            "propensity_category",
            "propensity_sku",
            "propensity_new_sku",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Synerise-like Retailrocket evaluation targets."
    )
    parser.add_argument("--data-dir", default="retailrocket_data")
    parser.add_argument("--out-dir", default="retailrocket_eval_full")
    parser.add_argument("--target-days", type=int, default=14)
    parser.add_argument("--min-history-events", type=int, default=5)
    parser.add_argument("--top-categories", type=int, default=5)
    parser.add_argument("--top-skus", type=int, default=5)
    parser.add_argument("--top-new-skus", type=int, default=5)
    parser.add_argument("--max-clients", type=int, default=0)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    input_dir = out_dir / "input"
    target_dir = out_dir / "target"

    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    events = load_retailrocket_events(data_dir)

    dataset_end = events["timestamp"].max()
    cutoff = dataset_end - pl.duration(days=args.target_days)

    # Polars expressions with Python datetime cutoff.
    cutoff_value = events.select(
        (pl.col("timestamp").max() - pl.duration(days=args.target_days)).alias(
            "cutoff"
        )
    )["cutoff"][0]

    print(f"Dataset end      : {dataset_end}")
    print(f"Observation cutoff: {cutoff_value}")
    print(f"Target window    : {cutoff_value} to {dataset_end}")

    history = events.filter(pl.col("timestamp") < cutoff_value)
    target = events.filter(pl.col("timestamp") >= cutoff_value)

    print(f"History events: {history.height:,}")
    print(f"Target events : {target.height:,}")

    # ------------------------------------------------------------
    # relevant_clients.npy
    # ------------------------------------------------------------
    relevant_clients_df = (
        history.group_by("client_id")
        .agg(pl.len().alias("history_events"))
        .filter(pl.col("history_events") >= args.min_history_events)
        .sort(["history_events", "client_id"], descending=[True, False])
    )

    if args.max_clients and args.max_clients > 0:
        relevant_clients_df = relevant_clients_df.head(args.max_clients)

    relevant_clients = (
        relevant_clients_df["client_id"].to_numpy().astype(np.int64)
    )

    if relevant_clients.size == 0:
        raise RuntimeError("No relevant clients found.")

    np.save(input_dir / "relevant_clients.npy", relevant_clients)

    print(f"Relevant clients: {len(relevant_clients):,}")

    relevant_set = set(map(int, relevant_clients.tolist()))

    history_relevant = history.filter(pl.col("client_id").is_in(relevant_clients))
    target_relevant = target.filter(pl.col("client_id").is_in(relevant_clients))

    history_purchases = history_relevant.filter(
        pl.col("event_type") == "product_buy"
    )
    target_purchases = target_relevant.filter(
        pl.col("event_type") == "product_buy"
    )

    history_buyer_ids = set(
        map(
            int,
            history_purchases["client_id"].unique().to_list(),
        )
    )

    target_buyer_ids = set(
        map(
            int,
            target_purchases["client_id"].unique().to_list(),
        )
    )

    target_active_ids = set(
        map(
            int,
            target_relevant["client_id"].unique().to_list(),
        )
    )

    active_clients = np.array(sorted(target_active_ids), dtype=np.int64)
    np.save(target_dir / "active_clients.npy", active_clients)

    # ------------------------------------------------------------
    # Propensity task IDs and popularity arrays
    # ------------------------------------------------------------
    prop_cat_ids, prop_cat_pop = top_ids_and_counts(
        history_purchases,
        id_col="category_id",
        count_name="purchase_count",
        top_n=args.top_categories,
    )

    prop_sku_ids, prop_sku_pop = top_ids_and_counts(
        history_purchases,
        id_col="sku",
        count_name="purchase_count",
        top_n=args.top_skus,
    )

    history_purchased_skus = set(
        map(int, history_purchases["sku"].drop_nulls().unique().to_list())
    )

    target_purchases = target_purchases.with_columns(
        pl.when(~pl.col("sku").is_in(list(history_purchased_skus)))
        .then(pl.col("sku"))
        .otherwise(None)
        .alias("new_sku")
    )

    prop_new_sku_ids, prop_new_sku_pop = top_ids_and_counts(
        target_purchases.filter(pl.col("new_sku").is_not_null()),
        id_col="new_sku",
        count_name="target_purchase_count",
        top_n=args.top_new_skus,
    )

    np.save(target_dir / "propensity_category.npy", prop_cat_ids)
    np.save(target_dir / "propensity_sku.npy", prop_sku_ids)
    np.save(target_dir / "propensity_new_sku.npy", prop_new_sku_ids)

    np.save(target_dir / "popularity_propensity_category.npy", prop_cat_pop)
    np.save(target_dir / "popularity_propensity_sku.npy", prop_sku_pop)
    np.save(
        target_dir / "popularity_propensity_new_sku.npy",
        prop_new_sku_pop,
    )

    # Retailrocket has no validated price target.
    # Empty files are saved only to preserve Synerise-like folder structure.
    np.save(target_dir / "propensity_price.npy", np.array([], dtype=np.int64))
    np.save(
        target_dir / "popularity_propensity_price.npy",
        np.array([], dtype=np.int64),
    )

    # ------------------------------------------------------------
    # Restrict target label lists to selected task IDs
    # ------------------------------------------------------------
    prop_cat_set = set(map(int, prop_cat_ids.tolist()))
    prop_sku_set = set(map(int, prop_sku_ids.tolist()))
    prop_new_sku_set = set(map(int, prop_new_sku_ids.tolist()))

    target_purchases_for_labels = target_purchases.with_columns(
        [
            pl.when(pl.col("category_id").is_in(list(prop_cat_set)))
            .then(pl.col("category_id"))
            .otherwise(None)
            .alias("category_id"),
            pl.when(pl.col("sku").is_in(list(prop_sku_set)))
            .then(pl.col("sku"))
            .otherwise(None)
            .alias("sku"),
            pl.when(pl.col("new_sku").is_in(list(prop_new_sku_set)))
            .then(pl.col("new_sku"))
            .otherwise(None)
            .alias("new_sku"),
        ]
    )

    # ------------------------------------------------------------
    # Train/validation split on client IDs
    # ------------------------------------------------------------
    rng = np.random.default_rng(args.seed)

    # Clients mit mindestens einem Propensity-Label (aus allen drei Spalten) identifizieren
    labeled_client_ids = set(
        target_purchases_for_labels
        .filter(
            pl.col("category_id").is_not_null()
            | pl.col("sku").is_not_null()
            | pl.col("new_sku").is_not_null()
        )["client_id"]
        .unique()
        .to_list()
    )
    
    labeled_mask = np.isin(relevant_clients, list(labeled_client_ids))
    labeled_clients = relevant_clients[labeled_mask]
    unlabeled_clients = relevant_clients[~labeled_mask]
    
    # Beide Gruppen GETRENNT shuffeln und im selben Verhältnis splitten
    rng.shuffle(labeled_clients)
    rng.shuffle(unlabeled_clients)

    n_valid_labeled = max(1, int(len(labeled_clients) * args.validation_ratio))
    n_valid_unlabeled = max(1, int(len(unlabeled_clients) * args.validation_ratio))
    
    valid_clients = np.sort(np.concatenate([
        labeled_clients[:n_valid_labeled],
        unlabeled_clients[:n_valid_unlabeled],
    ]))
    train_clients = np.sort(np.concatenate([
        labeled_clients[n_valid_labeled:],
        unlabeled_clients[n_valid_unlabeled:],
    ]))

    train_target = build_target_frame(
        clients=train_clients,
        history_buyers=history_buyer_ids,
        target_active_clients=target_active_ids,
        target_buyers=target_buyer_ids,
        target_purchases=target_purchases_for_labels.filter(
            pl.col("client_id").is_in(train_clients)
        ),
    )

    validation_target = build_target_frame(
        clients=valid_clients,
        history_buyers=history_buyer_ids,
        target_active_clients=target_active_ids,
        target_buyers=target_buyer_ids,
        target_purchases=target_purchases_for_labels.filter(
            pl.col("client_id").is_in(valid_clients)
        ),
    )

    train_target.write_parquet(target_dir / "train_target.parquet")
    validation_target.write_parquet(target_dir / "validation_target.parquet")

    print("\nSaved files:")
    print(f"- {input_dir / 'relevant_clients.npy'}")
    print(f"- {target_dir / 'train_target.parquet'}")
    print(f"- {target_dir / 'validation_target.parquet'}")
    print(f"- {target_dir / 'active_clients.npy'}")
    print(f"- {target_dir / 'propensity_category.npy'} {prop_cat_ids.shape}")
    print(f"- {target_dir / 'propensity_sku.npy'} {prop_sku_ids.shape}")
    print(f"- {target_dir / 'propensity_new_sku.npy'} {prop_new_sku_ids.shape}")

    print("\nLabel summary:")
    print(f"Train clients     : {train_target.height:,}")
    print(f"Validation clients: {validation_target.height:,}")
    print(f"Active clients    : {len(active_clients):,}")
    print(f"History buyers    : {len(history_buyer_ids):,}")
    print(f"Target buyers     : {len(target_buyer_ids):,}")

    print("\nTop category IDs:", prop_cat_ids.tolist())
    print("Top SKU IDs     :", prop_sku_ids.tolist())
    print("Top new SKU IDs :", prop_new_sku_ids.tolist())


if __name__ == "__main__":
    main()