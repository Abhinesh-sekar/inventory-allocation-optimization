"""Unit tests for baseline allocation heuristics (Proportional and Greedy)."""

import pandas as pd
import pytest
from src.models.baseline_heuristics import BaselineAllocators


@pytest.fixture
def small_network():
    stores_df = pd.DataFrame([
        {"store_id": "S1", "capacity": 50, "shipping_cost_per_unit": 1.0},
        {"store_id": "S2", "capacity": 50, "shipping_cost_per_unit": 2.0},
    ])
    skus_df = pd.DataFrame([
        {
            "sku_id": "SKU1",
            "retail_price": 100.0,
            "wholesale_cost": 40.0,
            "markdown_price": 50.0,
            "min_presentation": 5,
        },
    ])
    forecasts_df = pd.DataFrame([
        {"store_id": "S1", "sku_id": "SKU1", "forecast_demand": 30.0},
        {"store_id": "S2", "sku_id": "SKU1", "forecast_demand": 10.0},
    ])
    warehouse_df = pd.DataFrame([
        {"sku_id": "SKU1", "warehouse_stock": 40},
    ])
    return stores_df, skus_df, forecasts_df, warehouse_df


def test_proportional_allocation(small_network):
    stores_df, skus_df, forecasts_df, warehouse_df = small_network
    alloc_df = BaselineAllocators.proportional_allocation(stores_df, skus_df, forecasts_df, warehouse_df)

    assert len(alloc_df) == 2
    # S1 has 30/40 = 75% of demand -> 30 units
    # S2 has 10/40 = 25% of demand -> 10 units
    s1_alloc = alloc_df[alloc_df["store_id"] == "S1"]["allocated_units"].iloc[0]
    s2_alloc = alloc_df[alloc_df["store_id"] == "S2"]["allocated_units"].iloc[0]

    assert s1_alloc == 30
    assert s2_alloc == 10
    assert s1_alloc + s2_alloc <= 40


def test_greedy_marginal_heuristic(small_network):
    stores_df, skus_df, forecasts_df, warehouse_df = small_network
    alloc_df = BaselineAllocators.greedy_marginal_heuristic(stores_df, skus_df, forecasts_df, warehouse_df)

    assert len(alloc_df) == 2
    # Both stores should be allocated at least min_presentation (5 units)
    for _, row in alloc_df.iterrows():
        assert row["allocated_units"] >= 5
