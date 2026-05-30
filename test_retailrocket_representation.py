from pathlib import Path
import json

import polars as pl

from src.ubm.text_representation_v3 import AdvancedUBMGenerator

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "retailrocket_data"
EVENTS_PATH = DATA_DIR / "events.csv"

raw_events = pl.scan_csv(EVENTS_PATH)

# Select one visitor who definitely has a transaction.
client_id = (
    raw_events
    .filter(pl.col("event") == "transaction")
    .select("visitorid")
    .unique()
    .limit(1)
    .collect()["visitorid"][0]
)

print("Testing representation for visitor:", client_id)

generator = AdvancedUBMGenerator(
    data_dir=str(DATA_DIR),
    cache_dir=str(DATA_DIR / "cache_test"),
    debug_mode=True,
)

generator.load_data(
    use_cache=False,
    relevant_client_ids=[client_id],
)

representations = generator.generate_representations([client_id])

payload = json.loads(representations[client_id])

print("\nUser type:")
print(payload["profile"]["overview"]["user_type"])

print("\nReference time:")
print(generator.reference_time)

print("\nRich text representation:")
print(payload["rich_text"])

print("\nBehavioral metrics:")
print(payload["profile"]["behavioral_metrics"])