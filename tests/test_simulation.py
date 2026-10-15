"""Unit tests for Demand Realization & Markdown Simulation."""

import numpy as np
import pandas as pd
import pytest
from src.simulation.demand_simulator import DemandSimulator


@pytest.fixture
def simulation_data():
    stores_df = pd.DataFrame([{"store_id": "S1", "shipping_cost_per_unit": 1.0}])
    skus_df = pd.DataFrame([{
        "sku_id": "SKU1",
        "retail_price": 100.0,
        "wholesale_cost": 40.0,
        "markdown_price": 50.0,
        "min_presentation": 5,
    }])
    forecasts_df = pd.DataFrame([{"store_id": "S1", "sku_id": "SKU1", "forecast_demand": 20.0}])
    return stores_df, skus_df, forecasts_df


def test_simulation_single_trial(simulation_data):
    stores_df, skus_df, forecasts_df = simulation_data
    sim = DemandSimulator(stores_df, skus_df, forecasts_df)

    # Allocating 25 units when forecast is 20
    alloc_df = pd.DataFrame([{"store_id": "S1", "sku_id": "SKU1", "allocated_units": 25, "is_stocked": 1}])
    result = sim.simulate_single_trial(alloc_df, random_seed=42)

    assert "total_sales_units" in result
    assert "stockout_units" in result
    assert "markdown_cost" in result
    assert "net_profit" in result
    assert result["total_allocated_units"] == 25
    assert result["presentation_violations"] == 0


def test_presentation_violation(simulation_data):
    stores_df, skus_df, forecasts_df = simulation_data
    sim = DemandSimulator(stores_df, skus_df, forecasts_df)

    # Allocating 2 units (min_presentation is 5) -> 1 violation
    alloc_df = pd.DataFrame([{"store_id": "S1", "sku_id": "SKU1", "allocated_units": 2, "is_stocked": 1}])
    result = sim.simulate_single_trial(alloc_df, random_seed=42)

    assert result["presentation_violations"] == 1
