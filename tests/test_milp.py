"""Unit tests for Inventory MILP Optimizer."""

import numpy as np
import pandas as pd
import pytest
from src.models.milp_optimizer import InventoryMILPOptimizer


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
        {
            "sku_id": "SKU2",
            "retail_price": 60.0,
            "wholesale_cost": 25.0,
            "markdown_price": 30.0,
            "min_presentation": 4,
        },
    ])
    forecasts_df = pd.DataFrame([
        {"store_id": "S1", "sku_id": "SKU1", "forecast_demand": 20.0},
        {"store_id": "S1", "sku_id": "SKU2", "forecast_demand": 15.0},
        {"store_id": "S2", "sku_id": "SKU1", "forecast_demand": 25.0},
        {"store_id": "S2", "sku_id": "SKU2", "forecast_demand": 10.0},
    ])
    warehouse_df = pd.DataFrame([
        {"sku_id": "SKU1", "warehouse_stock": 35}, # Constrained: total demand 45
        {"sku_id": "SKU2", "warehouse_stock": 20}, # Constrained: total demand 25
    ])
    return stores_df, skus_df, forecasts_df, warehouse_df


def test_milp_optimizer_constraints(small_network):
    stores_df, skus_df, forecasts_df, warehouse_df = small_network
    optimizer = InventoryMILPOptimizer(time_limit_sec=10, gap_rel=0.01)

    alloc_df, diag = optimizer.solve(stores_df, skus_df, forecasts_df, warehouse_df)

    assert diag["solver_status"] in ["Optimal", "Not Solved"]
    assert len(alloc_df) == 4

    # 1. Warehouse stock bounds
    sku1_alloc = alloc_df[alloc_df["sku_id"] == "SKU1"]["allocated_units"].sum()
    sku2_alloc = alloc_df[alloc_df["sku_id"] == "SKU2"]["allocated_units"].sum()
    assert sku1_alloc <= 35
    assert sku2_alloc <= 20

    # 2. Store capacity bounds
    s1_alloc = alloc_df[alloc_df["store_id"] == "S1"]["allocated_units"].sum()
    s2_alloc = alloc_df[alloc_df["store_id"] == "S2"]["allocated_units"].sum()
    assert s1_alloc <= 50
    assert s2_alloc <= 50

    # 3. Minimum presentation stock: if allocated > 0, allocated >= min_presentation
    min_pres_map = {"SKU1": 5, "SKU2": 4}
    for _, row in alloc_df.iterrows():
        units = row["allocated_units"]
        if units > 0:
            assert units >= min_pres_map[row["sku_id"]]
