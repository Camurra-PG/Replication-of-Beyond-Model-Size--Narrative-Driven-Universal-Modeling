from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "retailrocket_data"

properties_paths = [
    DATA_DIR / "item_properties_part1.csv",
    DATA_DIR / "item_properties_part2.csv",
]

for path in properties_paths:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

properties = pl.concat(
    [
        pl.scan_csv(properties_paths[0]),
        pl.scan_csv(properties_paths[1]),
    ]
)

print("Checking Retailrocket property names...\n")

# Non-numeric names are the only properties that may be directly interpretable.
named_properties = (
    properties
    .filter(
        ~pl.col("property").str.contains(r"^\d+$")
    )
    .group_by("property")
    .agg(pl.len().alias("rows"))
    .sort("rows", descending=True)
    .collect(engine="streaming")
)

print("Non-numeric property names:")
for row in named_properties.to_dicts():
    print(row)

for property_name in ["categoryid", "available", "price"]:
    rows = (
        properties
        .filter(pl.col("property") == property_name)
        .select([
            "timestamp",
            "itemid",
            "property",
            "value",
        ])
        .limit(5)
        .collect(engine="streaming")
    )

    count = (
        properties
        .filter(pl.col("property") == property_name)
        .select(pl.len().alias("count"))
        .collect(engine="streaming")["count"][0]
    )

    print(f"\nProperty '{property_name}': {count:,} rows")
    for row in rows.to_dicts():
        print(row)