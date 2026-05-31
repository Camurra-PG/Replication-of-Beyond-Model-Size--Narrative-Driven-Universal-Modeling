from datetime import timedelta
from pathlib import Path
import json

import polars as pl

from src.ubm.text_representation_v3 import AdvancedUBMGenerator

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "retailrocket_data"
EVENTS_PATH = DATA_DIR / "events.csv"

# ------------------------------------------------------------
# Read Retailrocket events with proper timestamps
# ------------------------------------------------------------
raw_events = (
    pl.scan_csv(EVENTS_PATH)
    .with_columns(
        pl.from_epoch(
            pl.col("timestamp").cast(pl.Int64),
            time_unit="ms"
        ).alias("event_time")
    )
)

dataset_end = (
    raw_events
    .select(pl.col("event_time").max().alias("dataset_end"))
    .collect()["dataset_end"][0]
)

cutoff = dataset_end - timedelta(days=14)

print("Dataset end:", dataset_end)
print("Observation cutoff:", cutoff)
print("Target window:", cutoff, "to", dataset_end)

# ------------------------------------------------------------
# Find a user with sufficient history and a purchase afterwards
# ------------------------------------------------------------
history_users = (
    raw_events
    .filter(pl.col("event_time") < cutoff)
    .group_by("visitorid")
    .agg(pl.len().alias("history_events"))
    .filter(pl.col("history_events") >= 5)
)

target_buyers = (
    raw_events
    .filter(
        (pl.col("event_time") >= cutoff)
        & (pl.col("event_time") <= dataset_end)
        & (pl.col("event") == "transaction")
    )
    .select("visitorid")
    .unique()
)

eligible_users = (
    history_users
    .join(target_buyers, on="visitorid", how="inner")
    .sort("history_events", descending=True)
    .limit(1)
    .collect()
)

if eligible_users.is_empty():
    raise RuntimeError("No eligible temporal-split test user found.")

client_id = eligible_users["visitorid"][0]
print("Testing client:", client_id)

# ------------------------------------------------------------
# Build profile only from observation history
# ------------------------------------------------------------
generator = AdvancedUBMGenerator(
    data_dir=str(DATA_DIR),
    cache_dir=str(DATA_DIR / "cache_temporal_test"),
    debug_mode=True,
)

generator.load_data(
    use_cache=False,
    relevant_client_ids=[client_id],
    observation_end=cutoff,
)

history = generator.get_client_events(client_id)

print("\nReference time used by generator:")
print(generator.reference_time)

print("\nMaximum event time inside profile history:")
print(history["timestamp"].max())

print("\nHistory event count:")
print(history.height)

assert generator.reference_time == cutoff
assert history["timestamp"].max() < cutoff

representations = generator.generate_representations([client_id])
payload = json.loads(representations[client_id])

print("\nTemporal split verified: no target-window events entered the profile.")
print("\nProfile user type based only on history:")
print(payload["profile"]["overview"]["user_type"])

print("\nRich text preview:")
print(payload["rich_text"][:1500])