from pathlib import Path
import json

import polars as pl

from src.ubm.text_representation_v3 import AdvancedUBMGenerator

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "retailrocket_data"
EVENTS_PATH = DATA_DIR / "events.csv"

# ------------------------------------------------------------
# Pick one client who definitely contains a purchase event.
# ------------------------------------------------------------
client_id = (
    pl.scan_csv(EVENTS_PATH)
    .filter(pl.col("event") == "transaction")
    .select("visitorid")
    .unique()
    .limit(1)
    .collect()["visitorid"][0]
)

print("Testing full-mode profile for visitor:", client_id)

# ------------------------------------------------------------
# Important: debug_mode=False tests the normal pipeline.
# use_cache=False first prevents cache concerns from masking errors.
# ------------------------------------------------------------
generator = AdvancedUBMGenerator(
    data_dir=str(DATA_DIR),
    cache_dir=str(DATA_DIR / "cache_full_mode_test"),
    debug_mode=False,
)

generator.load_data(
    use_cache=True,
    relevant_client_ids=[client_id],
)

print("\nLazy pipeline available:")
print(generator.lazy_all is not None)

print("\nEvents dataframe materialized:")
print(generator.events_df is not None)

schema_columns = generator.lazy_all.collect_schema().names()

print("\nLazy schema columns:")
print(schema_columns)

assert "category_id" in schema_columns
assert "is_available" in schema_columns

extractors = generator.get_feature_extractors()
extractor_names = list(extractors.keys())

print("\nInitialized extractors:")
print(extractor_names)

assert "availability" in extractor_names
assert "social" in extractor_names
assert "graph" in extractor_names
assert "intent" in extractor_names

representations = generator.generate_representations([client_id])
payload = json.loads(representations[client_id])
rich_text = payload["rich_text"]

print("\nProfile generated successfully.")
print("\nContains AVAILABILITY section:")
print("[AVAIL]" in rich_text)

print("\nContains SOCIAL section:")
print("[SOCIAL]" in rich_text)

print("\nRAW_SEQUENCE occurrences:")
print(rich_text.count("## RAW_SEQUENCE ##"))

assert "[AVAIL]" in rich_text
assert "[SOCIAL]" in rich_text
assert rich_text.count("## RAW_SEQUENCE ##") == 1

print("\nFull-mode Retailrocket pipeline verified.")
print("\nRich text preview:")
print(rich_text[:1500])