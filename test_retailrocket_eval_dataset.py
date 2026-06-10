from pathlib import Path

import numpy as np
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT / "retailrocket_eval"
INPUT_DIR = ROOT / "input"
TARGET_DIR = ROOT / "target"


def load_npy(name: str) -> np.ndarray:
    path = TARGET_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


def main() -> None:
    relevant_clients_path = INPUT_DIR / "relevant_clients.npy"
    train_target_path = TARGET_DIR / "train_target.parquet"
    validation_target_path = TARGET_DIR / "validation_target.parquet"

    if not relevant_clients_path.exists():
        raise FileNotFoundError(f"Missing file: {relevant_clients_path}")

    if not train_target_path.exists():
        raise FileNotFoundError(f"Missing file: {train_target_path}")

    if not validation_target_path.exists():
        raise FileNotFoundError(f"Missing file: {validation_target_path}")

    relevant_clients = np.load(relevant_clients_path)
    active_clients = load_npy("active_clients.npy")

    propensity_category = load_npy("propensity_category.npy")
    propensity_sku = load_npy("propensity_sku.npy")
    propensity_new_sku = load_npy("propensity_new_sku.npy")

    popularity_category = load_npy("popularity_propensity_category.npy")
    popularity_sku = load_npy("popularity_propensity_sku.npy")
    popularity_new_sku = load_npy("popularity_propensity_new_sku.npy")

    propensity_price = load_npy("propensity_price.npy")
    popularity_price = load_npy("popularity_propensity_price.npy")

    train_target = pl.read_parquet(train_target_path)
    validation_target = pl.read_parquet(validation_target_path)

    print("=== NPY FILES ===")
    print("relevant_clients:", relevant_clients.shape, relevant_clients.dtype)
    print("active_clients:", active_clients.shape, active_clients.dtype)

    print("propensity_category:", propensity_category.shape, propensity_category.tolist())
    print("propensity_sku:", propensity_sku.shape, propensity_sku.tolist())
    print("propensity_new_sku:", propensity_new_sku.shape, propensity_new_sku.tolist())

    print("popularity_category:", popularity_category.shape, popularity_category.tolist())
    print("popularity_sku:", popularity_sku.shape, popularity_sku.tolist())
    print("popularity_new_sku:", popularity_new_sku.shape, popularity_new_sku.tolist())

    print("propensity_price:", propensity_price.shape, propensity_price.tolist())
    print("popularity_price:", popularity_price.shape, popularity_price.tolist())

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

    print("\n=== CONSISTENCY CHECKS ===")

    assert relevant_clients.ndim == 1
    assert relevant_clients.dtype == np.int64
    assert relevant_clients.shape[0] == 5000, (
        f"Expected 5000 relevant clients in debug setup, got {relevant_clients.shape[0]}"
    )

    assert train_target.height == 4000, (
        f"Expected 4000 train rows, got {train_target.height}"
    )
    assert validation_target.height == 1000, (
        f"Expected 1000 validation rows, got {validation_target.height}"
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

    assert propensity_category.shape[0] <= 5
    assert propensity_sku.shape[0] <= 5
    assert propensity_new_sku.shape[0] <= 5

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

    print("\nRetailrocket evaluation dataset verified successfully.")


if __name__ == "__main__":
    main()