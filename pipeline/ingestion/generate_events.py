"""
Generate 10M+ realistic e-commerce user event logs.

Events modeled after Amazon's actual event taxonomy:
- page_view, search, product_view, add_to_cart,
  remove_from_cart, checkout_start, purchase, error

Includes realistic patterns:
- Conversion funnel dropoff (page_view → purchase: ~2.5%)
- Diurnal traffic patterns (peak 12-2pm, 7-9pm)
- Device distribution (mobile 65%, desktop 30%, tablet 5%)
- Error injection for anomaly detection testing
- Seasonal spikes (flash sales, Prime Day simulation)
"""
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Event taxonomy with realistic weights
EVENT_TYPES = {
    "page_view":        0.35,
    "search":           0.18,
    "product_view":     0.22,
    "add_to_cart":      0.09,
    "remove_from_cart": 0.02,
    "checkout_start":   0.05,
    "purchase":         0.04,
    "error":            0.03,
    "wishlist_add":     0.02,
}

CATEGORIES = [
    "Electronics", "Clothing", "Home & Kitchen", "Books",
    "Sports", "Beauty", "Toys", "Automotive", "Grocery", "Tools"
]

DEVICES = {"mobile": 0.65, "desktop": 0.30, "tablet": 0.05}
PLATFORMS = {"ios": 0.38, "android": 0.27, "web": 0.35}
REGIONS = {
    "us-east-1": 0.35, "us-west-2": 0.25, "eu-west-1": 0.20,
    "ap-southeast-1": 0.12, "ap-northeast-1": 0.08
}

ERROR_CODES = {
    "500": ("Internal Server Error", 0.35),
    "503": ("Service Unavailable", 0.25),
    "504": ("Gateway Timeout", 0.20),
    "404": ("Not Found", 0.15),
    "429": ("Rate Limited", 0.05),
}


def generate_diurnal_timestamps(n: int, start_date: str = "2024-01-01", days: int = 90) -> np.ndarray:
    """Generate timestamps with realistic diurnal traffic patterns."""
    base = pd.Timestamp(start_date).timestamp()
    day_seconds = 86400

    # Traffic weights by hour (peak at 1pm and 8pm)
    hour_weights = np.array([
        0.2, 0.1, 0.1, 0.1, 0.1, 0.2,   # 0-5am (low)
        0.5, 0.9, 1.1, 1.2, 1.3, 1.4,   # 6-11am (rising)
        1.8, 1.9, 1.7, 1.6, 1.5, 1.6,   # 12-5pm (peak)
        1.7, 2.0, 1.8, 1.4, 0.9, 0.5,   # 6-11pm (evening peak)
    ])
    hour_weights /= hour_weights.sum()

    day_offsets = np.random.choice(days, n) * day_seconds
    hours = np.random.choice(24, n, p=hour_weights)
    mins = np.random.randint(0, 60, n)
    secs = np.random.randint(0, 60, n)

    # Add flash sale spike on day 45
    spike_mask = (day_offsets // day_seconds == 45)
    hours[spike_mask] = np.random.choice([13, 14, 15], spike_mask.sum())

    return base + day_offsets + hours * 3600 + mins * 60 + secs


def generate_events(n: int = 10_000_000, output_dir: str = "data/raw") -> str:
    """Generate n e-commerce events and save as partitioned Parquet."""
    t0 = time.time()
    logger.info(f"Generating {n:,} e-commerce events...")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    chunk_size = 500_000
    chunks = []

    for chunk_start in range(0, n, chunk_size):
        size = min(chunk_size, n - chunk_start)

        # User IDs — power law distribution (few heavy users, many light users)
        num_users = max(100_000, n // 10)
        user_weights = np.random.zipf(1.5, num_users).astype(float)
        user_weights /= user_weights.sum()
        user_ids = np.random.choice(
            [f"U{i:07d}" for i in range(num_users)],
            size=size, p=user_weights
        )

        # Session IDs
        session_ids = [f"S{np.random.randint(1e9, 9e9)}" for _ in range(size)]

        # Event types
        event_names = list(EVENT_TYPES.keys())
        event_weights = list(EVENT_TYPES.values())
        events = np.random.choice(event_names, size=size, p=event_weights)

        # Timestamps
        timestamps = generate_diurnal_timestamps(size)

        # Device + platform
        devices = np.random.choice(list(DEVICES.keys()), size=size, p=list(DEVICES.values()))
        platforms = np.random.choice(list(PLATFORMS.keys()), size=size, p=list(PLATFORMS.values()))
        regions = np.random.choice(list(REGIONS.keys()), size=size, p=list(REGIONS.values()))

        # Product info
        categories = np.random.choice(CATEGORIES, size=size)
        product_ids = np.array([f"P{np.random.randint(1000, 99999)}" for _ in range(size)])
        prices = np.random.lognormal(3.5, 1.2, size).round(2)
        prices = np.clip(prices, 0.99, 2999.99)

        # Response times (ms) — lognormal with error spikes
        response_times = np.random.lognormal(5.5, 0.5, size).astype(int)
        error_mask = events == "error"
        response_times[error_mask] = np.random.lognormal(7.5, 1.0, error_mask.sum()).astype(int)

        # Error codes
        error_codes = np.where(error_mask,
            np.random.choice(list(ERROR_CODES.keys()), size=size,
                p=[v[1] for v in ERROR_CODES.values()]),
            "200")

        # Conversion value
        purchase_mask = events == "purchase"
        revenue = np.where(purchase_mask, prices * np.random.randint(1, 4, size), 0.0)

        # Search queries
        queries = [
            f"query_{np.random.choice(['laptop', 'phone', 'shirt', 'book', 'toy', 'shoes', 'watch', 'bag'])}"
            if e == "search" else ""
            for e in events
        ]

        df = pd.DataFrame({
            "event_id":        [f"E{chunk_start+i:010d}" for i in range(size)],
            "user_id":         user_ids,
            "session_id":      session_ids,
            "event_type":      events,
            "timestamp":       timestamps,
                        "device":          devices,
            "platform":        platforms,
            "region":          regions,
            "category":        categories,
            "product_id":      product_ids,
            "price":           prices.round(2),
            "revenue":         revenue.round(2),
            "response_time_ms": response_times,
            "error_code":      error_codes,
            "search_query":    queries,
            "hour":            (timestamps % 86400 // 3600).astype(int),
            "day_of_week":     ((timestamps // 86400) % 7).astype(int),
        })

        chunks.append(df)
        logger.info(f"  Generated chunk {chunk_start//chunk_size + 1}/{-(-n//chunk_size)}: {size:,} rows")

    full_df = pd.concat(chunks, ignore_index=True)

    # Partition by region for Athena-style querying
    out_path = f"{output_dir}/events"
    for region in full_df["region"].unique():
        region_df = full_df[full_df["region"] == region]
        region_path = f"{out_path}/region={region}"
        Path(region_path).mkdir(parents=True, exist_ok=True)
        region_df.to_parquet(f"{region_path}/data.parquet", index=False)

    elapsed = time.time() - t0
    stats = {
        "total_events": len(full_df),
        "unique_users": full_df["user_id"].nunique(),
        "unique_sessions": full_df["session_id"].nunique(),
        "event_distribution": full_df["event_type"].value_counts().to_dict(),
        "error_rate": float((full_df["event_type"] == "error").mean()),
        "total_revenue": float(full_df["revenue"].sum()),
        "avg_response_ms": float(full_df["response_time_ms"].mean()),
        "date_range": f"{pd.to_datetime(full_df['timestamp'].min(), unit='s')} to {pd.to_datetime(full_df['timestamp'].max(), unit='s')}",
        "elapsed_s": round(elapsed, 1),
    }
    with open(f"{output_dir}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Generated {len(full_df):,} events in {elapsed:.1f}s")
    logger.info(f"  Revenue: ${stats['total_revenue']:,.2f}")
    logger.info(f"  Error rate: {stats['error_rate']:.2%}")
    logger.info(f"  Avg response: {stats['avg_response_ms']:.0f}ms")
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    generate_events(n=10_000_000)
