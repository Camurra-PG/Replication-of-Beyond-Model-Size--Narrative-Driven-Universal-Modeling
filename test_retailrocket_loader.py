from pathlib import Path

import polars as pl

from src.ubm.text_representation_v3 import AdvancedUBMGenerator

# Project root = folder in which this test file is located.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "retailrocket_data"

events_path = DATA_DIR / "events.csv"

print("Project root:", PROJECT_ROOT)
print("Data directory:", DATA_DIR)
print("Events file:", events_path)
print("Events file exists:", events_path.exists())

if not events_path.exists():
    raise FileNotFoundError(f"Retailrocket events.csv not found: {events_path}")

# Read a few real visitor IDs directly from Retailrocket.
sample_events = pl.read_csv(events_path).head(100)
test_client_ids = sample_events["visitorid"].unique().head(5).to_list()

print("Testing visitor IDs:", test_client_ids)

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

df = generator.lazy_all.sort(["client_id", "timestamp"]).collect()

print("\nColumns:")
print(df.columns)

print("\nShape:")
print(df.shape)

print("\nEvent types:")
print(df["event_type"].unique().to_list())

print("\nFirst rows as dictionaries:")
for row in df.head().to_dicts():
    print(row)


print("\nRows with category IDs:")
for row in df.select(
    ["client_id", "sku", "event_type", "category_id"]
).head().to_dicts():
    print(row)

print("\nEvents with known category:")
print(df.filter(pl.col("category_id").is_not_null()).height)