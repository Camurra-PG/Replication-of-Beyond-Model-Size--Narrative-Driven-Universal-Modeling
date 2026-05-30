from pathlib import Path

import polars as pl

from src.ubm.text_representation_v3 import AdvancedUBMGenerator

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "retailrocket_data"
EVENTS_PATH = DATA_DIR / "events.csv"

print("Project root:", PROJECT_ROOT)
print("Data directory:", DATA_DIR)
print("Events file:", EVENTS_PATH)
print("Events file exists:", EVENTS_PATH.exists())

if not EVENTS_PATH.exists():
    raise FileNotFoundError(f"Retailrocket events.csv not found: {EVENTS_PATH}")

# ------------------------------------------------------------
# Select users who definitely contain transaction events
# ------------------------------------------------------------
raw_events = pl.scan_csv(EVENTS_PATH)

transaction_clients = (
    raw_events
    .filter(pl.col("event") == "transaction")
    .select("visitorid")
    .unique()
    .limit(3)
    .collect()["visitorid"]
    .to_list()
)

cart_clients = (
    raw_events
    .filter(pl.col("event") == "addtocart")
    .select("visitorid")
    .unique()
    .limit(2)
    .collect()["visitorid"]
    .to_list()
)

test_client_ids = list(dict.fromkeys(transaction_clients + cart_clients))

print("Testing visitor IDs:", test_client_ids)
print("Transaction visitor IDs:", transaction_clients)
print("Cart visitor IDs:", cart_clients)

# ------------------------------------------------------------
# Load through the adapted Retailrocket loader
# ------------------------------------------------------------
generator = AdvancedUBMGenerator(
    data_dir=str(DATA_DIR),
    cache_dir=str(DATA_DIR / "cache_test"),
    debug_mode=True,
)

generator.load_data(
    use_cache=False,
    relevant_client_ids=test_client_ids,
)

print("\nLazy pipeline created successfully.")

df = (
    generator.lazy_all
    .sort(["client_id", "timestamp"])
    .collect()
)

# ------------------------------------------------------------
# Inspect schema and mapping results
# ------------------------------------------------------------
print("\nColumns:")
print(df.columns)

print("\nShape:")
print(df.shape)

print("\nEvent count by type:")
for row in (
    df.group_by("event_type")
      .agg(pl.len().alias("count"))
      .sort("event_type")
      .to_dicts()
):
    print(row)

print("\nFirst rows as dictionaries:")
for row in df.head(5).to_dicts():
    print(row)

print("\nPurchase rows:")
purchase_rows = (
    df.filter(pl.col("event_type") == "product_buy")
      .select(["client_id", "timestamp", "sku", "transaction_id", "category_id"])
      .head(10)
      .to_dicts()
)

for row in purchase_rows:
    print(row)

print("\nNumber of mapped purchase rows:")
print(len(df.filter(pl.col("event_type") == "product_buy")))

print("\nEvents with known category:")
print(df.filter(pl.col("category_id").is_not_null()).height)

print("\nReference time:")
print(generator.reference_time)