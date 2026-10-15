"""
Monte Carlo Demand Realization & Markdown Simulation Engine.
Simulates Poisson/Negative Binomial stochastic customer demand realizations across stores,
calculating realized sales, stock-out units, markdown liquidation costs, shipping fees, and net profit.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class DemandSimulator:
    """Simulates real-world retail demand realization and financial P&L."""

    def __init__(
        self,
        stores_df: pd.DataFrame,
        skus_df: pd.DataFrame,
        forecasts_df: pd.DataFrame,
    ):
        self.stores_df = stores_df.copy()
        self.skus_df = skus_df.copy()
        self.forecasts_df = forecasts_df.copy()

        # Lookup caches
        self.store_ship = dict(zip(stores_df["store_id"], stores_df["shipping_cost_per_unit"]))
        self.sku_price = dict(zip(skus_df["sku_id"], skus_df["retail_price"]))
        self.sku_cost = dict(zip(skus_df["sku_id"], skus_df["wholesale_cost"]))
        self.sku_mdown = dict(zip(skus_df["sku_id"], skus_df["markdown_price"]))
        self.sku_min_pres = dict(zip(skus_df["sku_id"], skus_df["min_presentation"]))
        self.dem_map = dict(zip(zip(forecasts_df["store_id"], forecasts_df["sku_id"]), forecasts_df["forecast_demand"]))

    def simulate_single_trial(
        self,
        allocations_df: pd.DataFrame,
        random_seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Runs one stochastic demand realization trial for an allocation plan.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        df = allocations_df.copy()
        df["forecast_demand"] = [self.dem_map.get((s, i), 0.0) for s, i in zip(df["store_id"], df["sku_id"])]
        df["retail_price"] = df["sku_id"].map(self.sku_price)
        df["wholesale_cost"] = df["sku_id"].map(self.sku_cost)
        df["markdown_price"] = df["sku_id"].map(self.sku_mdown)
        df["shipping_cost"] = df["store_id"].map(self.store_ship)
        df["min_presentation"] = df["sku_id"].map(self.sku_min_pres)

        # Realize stochastic demand via Poisson distribution parameterized by forecast mu
        df["actual_demand"] = np.random.poisson(df["forecast_demand"])

        # Operational metrics
        alloc = df["allocated_units"]
        dem = df["actual_demand"]

        # Presentation violations: 0 < alloc < min_presentation (broken size curve / visual display)
        df["presentation_violation"] = (alloc > 0) & (alloc < df["min_presentation"])

        # When a store receives fewer units than min_presentation, the broken size curve
        # drastically impairs customer discovery and conversion (orphan units rarely sell at full price)
        effective_conversion = np.where(
            df["presentation_violation"],
            np.maximum(0.20, (alloc / df["min_presentation"]) ** 1.5),
            1.0,
        )
        effective_demand = np.round(dem * effective_conversion).astype(int)

        df["sales_units"] = np.minimum(alloc, effective_demand)
        df["stockout_units"] = np.maximum(0, dem - df["sales_units"])
        df["leftover_units"] = np.maximum(0, alloc - df["sales_units"])

        # Financial P&L
        full_price_rev = (df["sales_units"] * df["retail_price"]).sum()
        markdown_rev = (df["leftover_units"] * df["markdown_price"]).sum()
        markdown_cost = (df["leftover_units"] * (df["retail_price"] - df["markdown_price"])).sum()
        cogs = (alloc * df["wholesale_cost"]).sum()
        shipping_total = (alloc * df["shipping_cost"]).sum()
        net_profit = full_price_rev + markdown_rev - cogs - shipping_total

        # Stockout on carried assortment (when a store actively carries a SKU but runs out of stock)
        df["carried_stockout_units"] = np.where(alloc > 0, np.maximum(0, dem - df["sales_units"]), 0)
        carried_demand = np.where(alloc > 0, dem, 0).sum()
        carried_stockouts = df["carried_stockout_units"].sum()
        carried_stockout_rate = (carried_stockouts / carried_demand * 100.0) if carried_demand > 0 else 0.0

        total_demand = dem.sum()
        total_sales = df["sales_units"].sum()
        total_stockout = df["stockout_units"].sum()
        total_leftover = df["leftover_units"].sum()
        total_allocated = alloc.sum()

        fill_rate = (total_sales / total_demand * 100.0) if total_demand > 0 else 0.0
        stockout_rate = (total_stockout / total_demand * 100.0) if total_demand > 0 else 0.0
        pres_violations = int(df["presentation_violation"].sum())

        return {
            "total_allocated_units": int(total_allocated),
            "total_demand_units": int(total_demand),
            "total_sales_units": int(total_sales),
            "stockout_units": int(total_stockout),
            "carried_stockout_units": int(carried_stockouts),
            "carried_stockout_rate_pct": float(carried_stockout_rate),
            "stockout_rate_pct": float(stockout_rate),
            "leftover_units": int(total_leftover),
            "markdown_cost": float(markdown_cost),
            "full_price_revenue": float(full_price_rev),
            "markdown_revenue": float(markdown_rev),
            "total_revenue": float(full_price_rev + markdown_rev),
            "shipping_cost": float(shipping_total),
            "cogs": float(cogs),
            "net_profit": float(net_profit),
            "fill_rate_pct": float(fill_rate),
            "presentation_violations": pres_violations,
        }

    def simulate_monte_carlo(
        self,
        allocations_df: pd.DataFrame,
        n_trials: int = 25,
        base_seed: int = 100,
    ) -> Dict[str, float]:
        """
        Runs Monte Carlo trials to estimate expectation and variance of metrics.
        """
        trial_metrics = []
        for trial_idx in range(n_trials):
            m = self.simulate_single_trial(allocations_df, random_seed=base_seed + trial_idx)
            trial_metrics.append(m)

        trials_df = pd.DataFrame(trial_metrics)
        summary = {}
        for col in trials_df.columns:
            summary[f"{col}_mean"] = float(trials_df[col].mean())
            summary[f"{col}_std"] = float(trials_df[col].std())

        return summary
