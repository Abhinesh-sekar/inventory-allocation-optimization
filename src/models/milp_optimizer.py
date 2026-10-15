"""
Store-Level Inventory Allocation Mixed-Integer Linear Program (MILP).
Formulated using PuLP (CBC Solver).
Maximizes net expected revenue subject to store physical capacity,
warehouse supply bounds, shipping cost trade-offs, and minimum presentation stock thresholds.
"""

import time
from typing import Dict, Optional, Tuple
import pandas as pd
import pulp


class InventoryMILPOptimizer:
    """
    Mixed-Integer Linear Programming (MILP) Inventory Allocator.
    """

    def __init__(
        self,
        time_limit_sec: int = 30,
        gap_rel: float = 0.01,
        verbose: bool = False,
    ):
        self.time_limit_sec = time_limit_sec
        self.gap_rel = gap_rel
        self.verbose = verbose
        self.problem: Optional[pulp.LpProblem] = None
        self.solver_status: str = "Unsolved"
        self.solve_time: float = 0.0

    def solve(
        self,
        stores_df: pd.DataFrame,
        skus_df: pd.DataFrame,
        forecasts_df: pd.DataFrame,
        warehouse_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Solves the store-level inventory allocation MILP.

        Returns:
            allocations_df: DataFrame with columns [store_id, sku_id, allocated_units, is_stocked]
            metrics: Solver diagnostics (status, solve_time, objective_val)
        """
        t0 = time.time()
        print(f"[MILP] Initializing PuLP optimization model for {len(stores_df)} stores and {len(skus_df)} SKUs...")

        prob = pulp.LpProblem("Inventory_Allocation_Optimization", pulp.LpMaximize)

        stores = stores_df["store_id"].tolist()
        skus = skus_df["sku_id"].tolist()

        # Build fast lookup maps
        store_cap = dict(zip(stores_df["store_id"], stores_df["capacity"]))
        store_ship = dict(zip(stores_df["store_id"], stores_df["shipping_cost_per_unit"]))
        
        sku_price = dict(zip(skus_df["sku_id"], skus_df["retail_price"]))
        sku_cost = dict(zip(skus_df["sku_id"], skus_df["wholesale_cost"]))
        sku_mdown = dict(zip(skus_df["sku_id"], skus_df["markdown_price"]))
        sku_min_pres = dict(zip(skus_df["sku_id"], skus_df["min_presentation"]))
        
        wh_stock = dict(zip(warehouse_df["sku_id"], warehouse_df["warehouse_stock"]))
        
        # Forecast demand lookup: (store, sku) -> demand
        dem_map = dict(zip(zip(forecasts_df["store_id"], forecasts_df["sku_id"]), forecasts_df["forecast_demand"]))

        # -------------------------------------------------------------
        # Decision Variables
        # -------------------------------------------------------------
        # x[s, i]: integer allocation of SKU i to store s
        # y[s, i]: binary trigger (1 if SKU i carried at store s, 0 otherwise)
        # v[s, i]: expected surplus inventory (sold at markdown discount)
        # u[s, i]: expected unmet demand (stockout penalty)
        x = {}
        y = {}
        v = {}
        u = {}

        for s in stores:
            for i in skus:
                ub = min(int(wh_stock.get(i, 0)), int(store_cap.get(s, 0)))
                x[s, i] = pulp.LpVariable(f"x_{s}_{i}", lowBound=0, upBound=ub, cat=pulp.LpInteger)
                y[s, i] = pulp.LpVariable(f"y_{s}_{i}", cat=pulp.LpBinary)
                v[s, i] = pulp.LpVariable(f"v_{s}_{i}", lowBound=0, cat=pulp.LpContinuous)
                u[s, i] = pulp.LpVariable(f"u_{s}_{i}", lowBound=0, cat=pulp.LpContinuous)

        # -------------------------------------------------------------
        # Objective Function:
        # Maximize Net Realized Margin:
        # r_full = P_i - Cost_i - Ship_s (full-price margin)
        # r_mdown = P_mdown - Cost_i - Ship_s (liquidation margin, typically negative/loss)
        # Expected Net Margin = r_full * (x - v) + r_mdown * v - stockout_pen * u
        #                     = r_full * x - (P_i - P_mdown) * v - stockout_pen * u
        # -------------------------------------------------------------
        obj_terms = []
        for s in stores:
            ship_cost = store_ship[s]
            for i in skus:
                price = sku_price[i]
                cost = sku_cost[i]
                mdown = sku_mdown[i]
                stockout_penalty = (price - cost) * 0.75  # severe stockout penalty (lost sales + customer goodwill)

                r_full = price - cost - ship_cost
                markdown_penalty = (price - mdown) + cost * 0.15  # markdown loss + holding cost

                obj_terms.append(r_full * x[s, i])
                obj_terms.append(-(r_full + markdown_penalty) * v[s, i])
                obj_terms.append(-stockout_penalty * u[s, i])

        prob += pulp.lpSum(obj_terms), "Net_Expected_Margin"

        # -------------------------------------------------------------
        # Constraints
        # -------------------------------------------------------------
        # 1. Warehouse Stock Bound per SKU
        for i in skus:
            avail = int(wh_stock.get(i, 0))
            prob += pulp.lpSum([x[s, i] for s in stores]) <= avail, f"Supply_Bound_{i}"

        # 2. Store Capacity Bound per Store
        for s in stores:
            cap = int(store_cap[s])
            prob += pulp.lpSum([x[s, i] for i in skus]) <= cap, f"Store_Capacity_{s}"

        # 3. Minimum Presentation Stock Linking & Assortment Hurdle:
        # x[s, i] >= min_pres * y[s, i]
        # x[s, i] <= U * y[s, i]
        # Assortment Hurdle: Only open SKU at store if forecasted demand justifies the presentation commitment (d >= min_pres)
        for s in stores:
            for i in skus:
                min_p = int(sku_min_pres[i])
                d = dem_map.get((s, i), 0.0)
                ub = min(int(wh_stock.get(i, 0)), int(store_cap[s]))

                if d < min_p:
                    # Low turnover store: do not carry SKU to avoid stranded markdown clearance
                    prob += y[s, i] == 0, f"Demand_Hurdle_{s}_{i}"
                    prob += x[s, i] == 0, f"Zero_Alloc_{s}_{i}"
                else:
                    prob += x[s, i] >= min_p * y[s, i], f"Min_Pres_{s}_{i}"
                    prob += x[s, i] <= ub * y[s, i], f"Max_Pres_{s}_{i}"

        # 4. Linearization of Surplus (Markdown) and Deficit (Stockout)
        # Uses Newsvendor critical fractile logic:
        # High-margin items carry a safety buffer to prevent stockouts (mu + 0.5*sigma)
        # Markdown-sensitive items carry lean stock to prevent markdowns (mu - 0.4*sigma)
        for s in stores:
            for i in skus:
                mu = dem_map.get((s, i), 0.0)
                margin_ratio = (sku_price[i] - sku_cost[i]) / sku_price[i]
                sigma = mu ** 0.5
                if margin_ratio >= 0.55:
                    target_d = mu + 0.5 * sigma
                elif margin_ratio <= 0.45:
                    target_d = max(0.0, mu - 0.4 * sigma)
                else:
                    target_d = mu

                prob += v[s, i] >= x[s, i] - target_d, f"Surplus_{s}_{i}"
                prob += u[s, i] >= target_d - x[s, i], f"Deficit_{s}_{i}"

        print(f"[MILP] Formulated problem with {len(prob.variables()):,} variables and {len(prob.constraints):,} constraints.")
        print(f"[MILP] Solving with CBC solver (time_limit={self.time_limit_sec}s, gap={self.gap_rel*100:.1f}%)...")

        solver = pulp.PULP_CBC_CMD(
            timeLimit=self.time_limit_sec,
            gapRel=self.gap_rel,
            msg=1 if self.verbose else 0,
        )
        prob.solve(solver)
        self.solve_time = time.time() - t0
        self.solver_status = pulp.LpStatus[prob.status]
        self.problem = prob

        print(f"[MILP] Solver finished in {self.solve_time:.2f}s. Status: {self.solver_status}. Objective: ${pulp.value(prob.objective):,.2f}")

        # Extract solution
        rows = []
        for s in stores:
            for i in skus:
                val = int(round(pulp.value(x[s, i]) or 0))
                is_stk = int(round(pulp.value(y[s, i]) or 0))
                rows.append({
                    "store_id": s,
                    "sku_id": i,
                    "allocated_units": val,
                    "is_stocked": is_stk,
                })

        allocations_df = pd.DataFrame(rows)
        diagnostics = {
            "solver_status": self.solver_status,
            "solve_time_sec": round(self.solve_time, 2),
            "objective_value": round(float(pulp.value(prob.objective)), 2),
            "total_allocated_units": int(allocations_df["allocated_units"].sum()),
        }
        return allocations_df, diagnostics
