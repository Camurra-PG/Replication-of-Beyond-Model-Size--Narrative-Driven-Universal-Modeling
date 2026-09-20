from pathlib import Path

import numpy as np
import polars as pl

import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT / "retailrocket_eval_full"
INPUT_DIR = ROOT / "input"
TARGET_DIR = ROOT / "target"


EXPECTED_CATEGORY_TARGETS = 100
EXPECTED_SKU_TARGETS = 100
EXPECTED_NEW_SKU_TARGETS = 20


def load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


def main() -> None:
    relevant_clients_path = INPUT_DIR / "relevant_clients.npy"
    train_target_path = TARGET_DIR / "train_target.parquet"
    validation_target_path = TARGET_DIR / "validation_target.parquet"

    relevant_clients = load_npy(relevant_clients_path)
    active_clients = load_npy(TARGET_DIR / "active_clients.npy")

    propensity_category = load_npy(TARGET_DIR / "propensity_category.npy")
    propensity_sku = load_npy(TARGET_DIR / "propensity_sku.npy")
    propensity_new_sku = load_npy(TARGET_DIR / "propensity_new_sku.npy")

    popularity_category = load_npy(TARGET_DIR / "popularity_propensity_category.npy")
    popularity_sku = load_npy(TARGET_DIR / "popularity_propensity_sku.npy")
    popularity_new_sku = load_npy(TARGET_DIR / "popularity_propensity_new_sku.npy")

    propensity_price = load_npy(TARGET_DIR / "propensity_price.npy")
    popularity_price = load_npy(TARGET_DIR / "popularity_propensity_price.npy")

    if not train_target_path.exists():
        raise FileNotFoundError(f"Missing file: {train_target_path}")
    if not validation_target_path.exists():
        raise FileNotFoundError(f"Missing file: {validation_target_path}")

    train_target = pl.read_parquet(train_target_path)
    validation_target = pl.read_parquet(validation_target_path)

    print("=== NPY FILES ===")
    print("relevant_clients:", relevant_clients.shape, relevant_clients.dtype)
    print("active_clients:", active_clients.shape, active_clients.dtype)

    print("propensity_category:", propensity_category.shape)
    print("propensity_sku:", propensity_sku.shape)
    print("propensity_new_sku:", propensity_new_sku.shape)

    print("popularity_category:", popularity_category.shape)
    print("popularity_sku:", popularity_sku.shape)
    print("popularity_new_sku:", popularity_new_sku.shape)

    print("propensity_price:", propensity_price.shape, propensity_price.tolist())
    print("popularity_price:", popularity_price.shape, popularity_price.tolist())

    print("\nTop 10 category targets:", propensity_category[:10].tolist())
    print("Top 10 sku targets:", propensity_sku[:10].tolist())
    print("Top new sku targets:", propensity_new_sku.tolist())

    print("\n=== PARQUET SCHEMA ===")
    print("train_target schema:")
    print(train_target.schema)

    print("\nvalidation_target schema:")
    print(validation_target.schema)

    print("\n=== TRAIN SAMPLE ===")
    print(train_target.head(10))

    print("\n=== VALIDATION SAMPLE ===")
    print(validation_target.head(10))

    print("\n=== LABEL SUMMARY ===")
    all_targets = pl.concat([train_target, validation_target])

    print("Total target rows:", all_targets.height)
    print("Train rows:", train_target.height)
    print("Validation rows:", validation_target.height)

    active_count = all_targets.filter(pl.col("active") == 1).height
    print("Active target clients:", active_count)

    churn_defined = all_targets.filter(pl.col("churn").is_not_null())
    churn_positive = churn_defined.filter(pl.col("churn") == 1).height
    churn_negative = churn_defined.filter(pl.col("churn") == 0).height

    print("Churn defined clients:", churn_defined.height)
    print("Churn positive:", churn_positive)
    print("Churn negative:", churn_negative)

    category_positive = all_targets.filter(
        pl.col("propensity_category").list.len() > 0
    ).height

    sku_positive = all_targets.filter(
        pl.col("propensity_sku").list.len() > 0
    ).height

    new_sku_positive = all_targets.filter(
        pl.col("propensity_new_sku").list.len() > 0
    ).height

    print("Clients with target category labels:", category_positive)
    print("Clients with target sku labels:", sku_positive)
    print("Clients with target new-sku labels:", new_sku_positive)

    print("\n=== TARGET LABEL FREQUENCIES ===")

    category_freq = (
        all_targets
        .select(pl.col("propensity_category").explode().alias("category_id"))
        .drop_nulls()
        .group_by("category_id")
        .agg(pl.len().alias("positive_clients"))
        .sort("positive_clients", descending=True)
    )

    sku_freq = (
        all_targets
        .select(pl.col("propensity_sku").explode().alias("sku"))
        .drop_nulls()
        .group_by("sku")
        .agg(pl.len().alias("positive_clients"))
        .sort("positive_clients", descending=True)
    )

    new_sku_freq = (
        all_targets
        .select(pl.col("propensity_new_sku").explode().alias("new_sku"))
        .drop_nulls()
        .group_by("new_sku")
        .agg(pl.len().alias("positive_clients"))
        .sort("positive_clients", descending=True)
    )

    print("\nCategory label frequency head:")
    print(category_freq.head(10))

    print("\nSKU label frequency head:")
    print(sku_freq.head(10))

    print("\nNew-SKU label frequency head:")
    print(new_sku_freq.head(10))

    print("\n=== CONSISTENCY CHECKS ===")

    assert relevant_clients.ndim == 1
    assert relevant_clients.dtype == np.int64
    assert relevant_clients.shape[0] > 5000, (
        f"Expected full setup to contain more than 5000 clients, got {relevant_clients.shape[0]}"
    )

    assert train_target.height > 0
    assert validation_target.height > 0
    assert train_target.height + validation_target.height == relevant_clients.shape[0], (
        "Train + validation row count does not match relevant_clients.npy."
    )

    expected_columns = {
        "client_id",
        "churn",
        "active",
        "propensity_category",
        "propensity_sku",
        "propensity_new_sku",
    }

    assert expected_columns.issubset(set(train_target.columns)), (
        f"Missing columns in train_target: {expected_columns - set(train_target.columns)}"
    )

    assert expected_columns.issubset(set(validation_target.columns)), (
        f"Missing columns in validation_target: {expected_columns - set(validation_target.columns)}"
    )

    assert propensity_category.shape == popularity_category.shape
    assert propensity_sku.shape == popularity_sku.shape
    assert propensity_new_sku.shape == popularity_new_sku.shape

    assert propensity_category.shape[0] <= EXPECTED_CATEGORY_TARGETS
    assert propensity_sku.shape[0] <= EXPECTED_SKU_TARGETS
    assert propensity_new_sku.shape[0] <= EXPECTED_NEW_SKU_TARGETS

    assert propensity_category.shape[0] > 0, "No category targets found."
    assert propensity_sku.shape[0] > 0, "No SKU targets found."
    assert propensity_new_sku.shape[0] > 0, "No new-SKU targets found."

    assert propensity_price.shape[0] == 0
    assert popularity_price.shape[0] == 0

    train_ids = set(train_target["client_id"].to_list())
    validation_ids = set(validation_target["client_id"].to_list())
    relevant_ids = set(relevant_clients.tolist())

    assert train_ids.isdisjoint(validation_ids), (
        "Train and validation clients overlap."
    )

    assert train_ids.union(validation_ids) == relevant_ids, (
        "Train + validation clients do not match relevant_clients.npy."
    )

    active_ids = set(active_clients.tolist())
    assert active_ids.issubset(relevant_ids), (
        "active_clients.npy contains clients not present in relevant_clients.npy."
    )

    assert active_count == len(active_clients), (
        "active_clients.npy count does not match active labels in target parquet files."
    )

    assert churn_defined.height > 0, "No churn labels defined."
    assert churn_positive > 0, "No positive churn labels."
    assert churn_negative > 0, "No negative churn labels."

    print("\nRetailrocket FULL evaluation dataset verified successfully.")


if __name__ == "__main__":
    main()