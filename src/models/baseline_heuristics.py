"""
Baseline Inventory Allocation Heuristics.
1. Proportional Demand Allocation (Industry baseline)
2. Greedy Marginal Profit Heuristic
Used to benchmark against the MILP optimization model.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


class BaselineAllocators:
    """Implements rule-based and greedy allocation benchmarks."""

    @staticmethod
    def proportional_allocation(
        stores_df: pd.DataFrame,
        skus_df: pd.DataFrame,
        forecasts_df: pd.DataFrame,
        warehouse_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Allocates warehouse stock strictly proportional to store forecasted demand.
        Ignores minimum presentation constraints and shipping cost trade-offs.
        Enforces physical store capacity by proportional clipping if exceeded.
        """
        print("[Baseline] Computing Proportional Demand Allocation Baseline...")
        wh_map = dict(zip(warehouse_df["sku_id"], warehouse_df["warehouse_stock"]))
        store_cap = dict(zip(stores_df["store_id"], stores_df["capacity"]))

        allocations = []
        for i_id, group in forecasts_df.groupby("sku_id"):
            avail_w = int(wh_map.get(i_id, 0))
            tot_d = group["forecast_demand"].sum()
            if tot_d <= 0 or avail_w <= 0:
                for s_id in group["store_id"]:
                    allocations.append({"store_id": s_id, "sku_id": i_id, "allocated_units": 0, "is_stocked": 0})
                continue

            shares = group["forecast_demand"] / tot_d
            exact_alloc = shares * avail_w
            floored = np.floor(exact_alloc).astype(int)
            remainder = int(avail_w - floored.sum())

            # Distribute remainder to largest fractional parts
            fractional = exact_alloc - floored
            if remainder > 0:
                top_rem_idx = fractional.nlargest(remainder).index
                floored.loc[top_rem_idx] += 1

            for s_id, alloc_val in zip(group["store_id"], floored):
                allocations.append({
                    "store_id": s_id,
                    "sku_id": i_id,
                    "allocated_units": int(alloc_val),
                    "is_stocked": 1 if alloc_val > 0 else 0,
                })

        alloc_df = pd.DataFrame(allocations)

        # Enforce store capacity constraint if total store units exceed capacity
        store_totals = alloc_df.groupby("store_id")["allocated_units"].sum()
        for s_id, tot in store_totals.items():
            cap = store_cap[s_id]
            if tot > cap:
                ratio = cap / tot
                mask = alloc_df["store_id"] == s_id
                alloc_df.loc[mask, "allocated_units"] = np.floor(alloc_df.loc[mask, "allocated_units"] * ratio).astype(int)
                alloc_df.loc[mask, "is_stocked"] = (alloc_df.loc[mask, "allocated_units"] > 0).astype(int)

        print(f"[Baseline] Proportional allocated {alloc_df['allocated_units'].sum():,} total units.")
        return alloc_df

    @staticmethod
    def greedy_marginal_heuristic(
        stores_df: pd.DataFrame,
        skus_df: pd.DataFrame,
        forecasts_df: pd.DataFrame,
        warehouse_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Greedy Priority Allocation:
        Sorts (store, SKU) pairs by expected net marginal profit = (retail_price - shipping_cost).
        First assigns minimum presentation stock to high-priority pairs, then tops up
        to forecasted demand until warehouse supply or store capacity is exhausted.
        """
        print("[Baseline] Computing Greedy Marginal Profit Allocation...")
        # Prepare working copies of capacities and warehouse inventory
        rem_cap = dict(zip(stores_df["store_id"], stores_df["capacity"].astype(int)))
        rem_wh = dict(zip(warehouse_df["sku_id"], warehouse_df["warehouse_stock"].astype(int)))
        
        store_ship = dict(zip(stores_df["store_id"], stores_df["shipping_cost_per_unit"]))
        sku_price = dict(zip(skus_df["sku_id"], skus_df["retail_price"]))
        sku_min_pres = dict(zip(skus_df["sku_id"], skus_df["min_presentation"]))

        # Compute priority ranking for each (store, SKU)
        df_work = forecasts_df.copy()
        df_work["retail_price"] = df_work["sku_id"].map(sku_price)
        df_work["shipping_cost"] = df_work["store_id"].map(store_ship)
        df_work["min_presentation"] = df_work["sku_id"].map(sku_min_pres)
        df_work["net_margin_per_unit"] = df_work["retail_price"] - df_work["shipping_cost"]

        # Sort descending by net margin per unit
        df_work = df_work.sort_values(by="net_margin_per_unit", ascending=False)

        allocated_map = {}

        # Pass 1: Try to satisfy minimum presentation stock for top items
        for _, row in df_work.iterrows():
            s = row["store_id"]
            i = row["sku_id"]
            d = int(np.round(row["forecast_demand"]))
            min_p = int(row["min_presentation"])

            if rem_cap[s] >= min_p and rem_wh[i] >= min_p:
                # We can allocate at least min_presentation
                desired = min(d, rem_cap[s], rem_wh[i])
                alloc = max(min_p, desired)
                allocated_map[(s, i)] = alloc
                rem_cap[s] -= alloc
                rem_wh[i] -= alloc
            else:
                allocated_map[(s, i)] = 0

        # Build output dataframe
        rows = []
        for _, row in forecasts_df.iterrows():
            s = row["store_id"]
            i = row["sku_id"]
            val = allocated_map.get((s, i), 0)
            rows.append({
                "store_id": s,
                "sku_id": i,
                "allocated_units": val,
                "is_stocked": 1 if val > 0 else 0,
            })

        alloc_df = pd.DataFrame(rows)
        print(f"[Baseline] Greedy allocated {alloc_df['allocated_units'].sum():,} total units.")
        return alloc_df
