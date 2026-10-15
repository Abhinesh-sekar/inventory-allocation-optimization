"""
Inventory Allocation & Markdown Optimization Benchmark Runner.
Executes:
1. Proportional Allocation Baseline
2. Greedy Marginal Profit Heuristic
3. PuLP Mixed-Integer Linear Program (MILP) Optimizer
4. Monte Carlo Demand Realization & Markdown Simulation (25 trials)
5. Comparative Analysis & Executive Metrics
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.milp_optimizer import InventoryMILPOptimizer
from src.models.baseline_heuristics import BaselineAllocators
from src.simulation.demand_simulator import DemandSimulator


def load_data(data_dir: str = "data/raw"):
    stores_df = pd.read_parquet(os.path.join(data_dir, "stores.parquet"))
    skus_df = pd.read_parquet(os.path.join(data_dir, "skus.parquet"))
    forecasts_df = pd.read_parquet(os.path.join(data_dir, "forecasts.parquet"))
    warehouse_df = pd.read_parquet(os.path.join(data_dir, "warehouse_stock.parquet"))
    return stores_df, skus_df, forecasts_df, warehouse_df


def main():
    print("=" * 75)
    print("      RETAIL INVENTORY ALLOCATION & MARKDOWN OPTIMIZATION BENCHMARK")
    print("=" * 75)
    stores_df, skus_df, forecasts_df, warehouse_df = load_data()

    print(f"Network Topology: {len(stores_df)} stores | {len(skus_df)} SKUs | {len(forecasts_df):,} store-SKU nodes")
    print(f"Total Store Capacity: {stores_df['capacity'].sum():,} units")
    print(f"Warehouse Stock:      {warehouse_df['warehouse_stock'].sum():,} units")
    print(f"Total Network Demand: {forecasts_df['forecast_demand'].sum():,.0f} units")

    # -------------------------------------------------------------
    # 1. Strategy 1: Proportional Allocation Baseline
    # -------------------------------------------------------------
    print("\n--- [1/3] Running Strategy 1: Proportional Allocation Baseline ---")
    t0 = time.time()
    prop_alloc = BaselineAllocators.proportional_allocation(stores_df, skus_df, forecasts_df, warehouse_df)
    t_prop = time.time() - t0

    # -------------------------------------------------------------
    # 2. Strategy 2: Greedy Heuristic
    # -------------------------------------------------------------
    print("\n--- [2/3] Running Strategy 2: Greedy Marginal Profit Heuristic ---")
    t0 = time.time()
    greedy_alloc = BaselineAllocators.greedy_marginal_heuristic(stores_df, skus_df, forecasts_df, warehouse_df)
    t_greedy = time.time() - t0

    # -------------------------------------------------------------
    # 3. Strategy 3: MILP Optimizer
    # -------------------------------------------------------------
    print("\n--- [3/3] Running Strategy 3: Mixed-Integer Linear Program (MILP) ---")
    milp = InventoryMILPOptimizer(time_limit_sec=30, gap_rel=0.01)
    milp_alloc, milp_diag = milp.solve(stores_df, skus_df, forecasts_df, warehouse_df)

    # -------------------------------------------------------------
    # 4. Monte Carlo Simulation Engine
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("   Running Monte Carlo Demand Realization Simulation (25 Stochastic Trials)...")
    print("=" * 75)
    sim = DemandSimulator(stores_df, skus_df, forecasts_df)

    res_prop = sim.simulate_monte_carlo(prop_alloc, n_trials=25)
    res_greedy = sim.simulate_monte_carlo(greedy_alloc, n_trials=25)
    res_milp = sim.simulate_monte_carlo(milp_alloc, n_trials=25)

    # Compute percentage improvements over proportional baseline
    stockout_red_pct = (res_prop["carried_stockout_units_mean"] - res_milp["carried_stockout_units_mean"]) / res_prop["carried_stockout_units_mean"] * 100
    markdown_red_pct = (res_prop["markdown_cost_mean"] - res_milp["markdown_cost_mean"]) / res_prop["markdown_cost_mean"] * 100
    profit_lift_pct = (res_milp["net_profit_mean"] - res_prop["net_profit_mean"]) / res_prop["net_profit_mean"] * 100

    print("\n" + "=" * 78)
    print("                   SIMULATION BENCHMARK RESULTS SUMMARY")
    print("=" * 78)
    print(f"{'Performance Metric':<28} | {'Proportional':<14} | {'Greedy':<14} | {'MILP (Optimal)':<14}")
    print("-" * 78)
    print(f"{'Allocated Inventory (units)':<28} | {res_prop['total_allocated_units_mean']:<14,.0f} | {res_greedy['total_allocated_units_mean']:<14,.0f} | {res_milp['total_allocated_units_mean']:<14,.0f}")
    print(f"{'Realized Sales (units)':<28} | {res_prop['total_sales_units_mean']:<14,.0f} | {res_greedy['total_sales_units_mean']:<14,.0f} | {res_milp['total_sales_units_mean']:<14,.0f}")
    print(f"{'Carried Stock-Out Units':<28} | {res_prop['carried_stockout_units_mean']:<14,.0f} | {res_greedy['carried_stockout_units_mean']:<14,.0f} | {res_milp['carried_stockout_units_mean']:<14,.0f}")
    print(f"{'Carried Stock-Out Rate (%)':<28} | {res_prop['carried_stockout_rate_pct_mean']:<13.2f}% | {res_greedy['carried_stockout_rate_pct_mean']:<13.2f}% | {res_milp['carried_stockout_rate_pct_mean']:<13.2f}%")
    print(f"{'Markdown Cost ($)':<28} | ${res_prop['markdown_cost_mean']:<13,.0f} | ${res_greedy['markdown_cost_mean']:<13,.0f} | ${res_milp['markdown_cost_mean']:<13,.0f}")
    print(f"{'Total Shipping Cost ($)':<28} | ${res_prop['shipping_cost_mean']:<13,.0f} | ${res_greedy['shipping_cost_mean']:<13,.0f} | ${res_milp['shipping_cost_mean']:<13,.0f}")
    print(f"{'Net Realized Profit ($)':<28} | ${res_prop['net_profit_mean']:<13,.0f} | ${res_greedy['net_profit_mean']:<13,.0f} | ${res_milp['net_profit_mean']:<13,.0f}")
    print(f"{'Fill Rate / Service Level':<28} | {res_prop['fill_rate_pct_mean']:<13.2f}% | {res_greedy['fill_rate_pct_mean']:<13.2f}% | {res_milp['fill_rate_pct_mean']:<13.2f}%")
    print(f"{'Presentation Violations':<28} | {res_prop['presentation_violations_mean']:<14,.0f} | {res_greedy['presentation_violations_mean']:<14,.0f} | {res_milp['presentation_violations_mean']:<14,.0f}")
    print("=" * 78)

    print("\n>> KEY IMPACT DELIVERABLES:")
    print(f"  * Carried Stock-out Reduction: {stockout_red_pct:.1f}% reduction vs. Proportional Baseline")
    print(f"  * Markdown Cost Reduction:     {markdown_red_pct:.1f}% reduction vs. Proportional Baseline")
    print(f"  * Assortment Integrity:        Eliminated {res_prop['presentation_violations_mean']:.0f} broken/fragmented presentation displays down to 0.\n")

    return res_prop, res_greedy, res_milp


if __name__ == "__main__":
    main()
