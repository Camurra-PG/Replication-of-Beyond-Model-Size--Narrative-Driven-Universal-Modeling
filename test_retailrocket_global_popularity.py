import json
from pathlib import Path

import polars as pl

from src.ubm.text_representation_v3 import (
    AdvancedUBMGenerator,
    TOP_RETAILROCKET_CATEGORIES,
    TOP_RETAILROCKET_SKUS,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "retailrocket_data"
CACHE_DIR = DATA_DIR / "cache_full_mode_test"


generator = AdvancedUBMGenerator(
    data_dir=str(DATA_DIR),
    cache_dir=str(CACHE_DIR),
    debug_mode=False,
)

generator.load_data(
    use_cache=True,
    relevant_client_ids=None,
)

assert generator.lazy_all is not None
assert generator.product_popularity is not None
assert generator.category_popularity is not None

top_skus = (
    generator.product_popularity
    .sort("popularity_score", descending=True)
    .head(TOP_RETAILROCKET_SKUS)["sku"]
    .drop_nulls()
    .to_list()
)

top_categories = (
    generator.category_popularity
    .sort("category_popularity_score", descending=True)
    .head(TOP_RETAILROCKET_CATEGORIES)["category_id"]
    .drop_nulls()
    .to_list()
)

candidate = (
    generator.lazy_all
    .filter(
        pl.col("client_id").is_not_null()
        & (
            pl.col("sku").is_in(top_skus)
            | pl.col("category_id").is_in(top_categories)
        )
    )
    .group_by("client_id")
    .agg([
        pl.len().alias("matching_events"),
        pl.col("sku").is_in(top_skus).sum().alias("top_sku_events"),
        pl.col("category_id").is_in(top_categories).sum().alias(
            "top_category_events"
        ),
    ])
    .sort("matching_events", descending=True)
    .head(1)
    .collect(engine="streaming")
)

assert candidate.height == 1, "No user with global top-item overlap found."

client_id = int(candidate["client_id"][0])

print(f"Testing client with global popularity overlap: {client_id}")
print(candidate)

representation = generator.generate_representations([client_id])
payload = json.loads(representation[client_id])
rich_text = payload["rich_text"]

print("\nRich text representation:\n")
print(rich_text)

assert "[TOP]" in rich_text, (
    "Expected a GLOBAL_POPULARITY section for a client selected "
    "from global top SKU/category interactions."
)

assert "GLOBAL_TOP_" in rich_text, (
    "Expected at least one global popularity feature."
)

print("\nGlobal Retailrocket popularity features verified.")