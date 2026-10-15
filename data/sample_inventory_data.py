"""
Synthetic Retail Network & Inventory Data Generator.
Generates store tiers, SKU catalog, central warehouse inventory,
and store-level demand forecasts with realistic retail variance.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd


def generate_inventory_dataset(
    n_stores: int = 25,
    n_skus: int = 100,
    random_seed: int = 42,
    output_dir: str = "data/raw",
):
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Stores Generator
    tiers = ["Flagship"] * 3 + ["High-Street"] * 7 + ["Mall"] * 10 + ["Outlet"] * 5
    tier_capacities = {
        "Flagship": (3500, 4500),
        "High-Street": (2200, 2800),
        "Mall": (1600, 2000),
        "Outlet": (1000, 1400),
    }
    tier_shipping = {
        "Flagship": 1.0,
        "High-Street": 1.25,
        "Mall": 1.5,
        "Outlet": 1.8,
    }

    stores = []
    for s_idx in range(n_stores):
        tier = tiers[s_idx]
        cap_low, cap_high = tier_capacities[tier]
        capacity = int(np.random.randint(cap_low, cap_high + 1))
        ship_mult = tier_shipping[tier]
        stores.append({
            "store_id": f"STORE_{s_idx+1:03d}",
            "store_name": f"{tier} Store {s_idx+1}",
            "store_tier": tier,
            "capacity": capacity,
            "shipping_cost_per_unit": round(1.20 * ship_mult, 2),
        })
    stores_df = pd.DataFrame(stores)

    # 2. SKUs Catalog Generator
    categories = ["Outerwear", "Dresses", "Tops", "Bottoms", "Footwear"]
    cat_price_ranges = {
        "Outerwear": (90, 180),
        "Dresses": (50, 120),
        "Tops": (25, 65),
        "Bottoms": (40, 85),
        "Footwear": (60, 130),
    }

    skus = []
    for i_idx in range(n_skus):
        cat = np.random.choice(categories, p=[0.15, 0.25, 0.30, 0.20, 0.10])
        p_min, p_max = cat_price_ranges[cat]
        price = round(float(np.random.uniform(p_min, p_max)), 2)
        cost = round(price * float(np.random.uniform(0.35, 0.45)), 2)
        markdown_price = round(price * float(np.random.uniform(0.40, 0.55)), 2)
        min_pres = int(np.random.choice([4, 5, 6, 8], p=[0.3, 0.4, 0.2, 0.1]))

        skus.append({
            "sku_id": f"SKU_{i_idx+1:04d}",
            "category": cat,
            "retail_price": price,
            "wholesale_cost": cost,
            "markdown_price": markdown_price,
            "min_presentation": min_pres,
            "margin": round(price - cost, 2),
        })
    skus_df = pd.DataFrame(skus)

    # 3. Store-Level Demand Forecasts Generator
    # Base SKU popularity follows a Pareto/lognormal curve
    sku_base_demand = np.random.lognormal(mean=2.8, sigma=0.6, size=n_skus)
    tier_demand_mult = {"Flagship": 1.6, "High-Street": 1.1, "Mall": 0.8, "Outlet": 0.5}

    forecast_rows = []
    for _, store in stores_df.iterrows():
        s_id = store["store_id"]
        t_mult = tier_demand_mult[store["store_tier"]]

        for idx, sku in skus_df.iterrows():
            i_id = sku["sku_id"]
            # Store-specific SKU affinity
            affinity = np.random.uniform(0.6, 1.4)
            mu = max(1.0, float(sku_base_demand[idx] * t_mult * affinity))
            sigma = round(float(mu * np.random.uniform(0.15, 0.35)), 2)
            forecast_rows.append({
                "store_id": s_id,
                "sku_id": i_id,
                "forecast_demand": round(mu, 2),
                "demand_std": sigma,
            })
    forecasts_df = pd.DataFrame(forecast_rows)

    # 4. Warehouse Inventory Stock
    # Realistic retail inventory mix:
    # - 40% high-velocity fast movers (demand > stock: coverage 0.70-0.85 -> stockout risk)
    # - 30% balanced staples (coverage 0.95-1.05)
    # - 30% slow movers / seasonal long tail (stock > demand: coverage 1.25-1.50 -> markdown liquidation risk)
    sku_total_demand = forecasts_df.groupby("sku_id")["forecast_demand"].sum()
    warehouse_stock = []
    for idx, (sku_id, total_dem) in enumerate(sku_total_demand.items()):
        if idx % 10 < 4:
            coverage = float(np.random.uniform(0.70, 0.85))
        elif idx % 10 < 7:
            coverage = float(np.random.uniform(0.95, 1.05))
        else:
            coverage = float(np.random.uniform(1.25, 1.50))

        avail = int(np.round(total_dem * coverage))
        warehouse_stock.append({
            "sku_id": sku_id,
            "warehouse_stock": max(10, avail),
            "total_network_demand": round(float(total_dem), 1),
            "stock_coverage_ratio": round(coverage, 2),
        })
    warehouse_df = pd.DataFrame(warehouse_stock)

    # Save to parquet
    stores_df.to_parquet(os.path.join(output_dir, "stores.parquet"))
    skus_df.to_parquet(os.path.join(output_dir, "skus.parquet"))
    forecasts_df.to_parquet(os.path.join(output_dir, "forecasts.parquet"))
    warehouse_df.to_parquet(os.path.join(output_dir, "warehouse_stock.parquet"))

    print(f"Generated Inventory Network:")
    print(f"  Stores: {len(stores_df)} (Total Capacity: {stores_df['capacity'].sum():,} units)")
    print(f"  SKUs: {len(skus_df)} catalog items")
    print(f"  Warehouse Inventory: {warehouse_df['warehouse_stock'].sum():,} units available")
    print(f"  Total Network Forecast Demand: {forecasts_df['forecast_demand'].sum():,.0f} units")
    return stores_df, skus_df, forecasts_df, warehouse_df


if __name__ == "__main__":
    generate_inventory_dataset()
