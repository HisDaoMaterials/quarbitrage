# Triple Barrier Labeling

from typing import Union
import polars as pl
from polars import DataFrame, LazyFrame

from quarbitrage.data_engineering.plugins import create_barrier_label_plugin


def create_triple_barrier_labels(
    df: Union[DataFrame, LazyFrame],
    price_col: str,
    volatility_col: str,
    partition_by: list[str] = None,
    order_by: list[str] = None,
    pt_multiplier: float | None = 1.0,
    sl_multiplier: float | None = 1.0,
    vertical_barrier_window: int | None = None,
    min_return: float | None = None,
) -> Union[DataFrame, LazyFrame]:
    """
    Create triple barrier labels with the following value breakdown:
       -1: If the lower horizontal barrier is hit first.
        1: If the upper horizontal barrier is hit first.
        0: If the vertical barrier is hit first and final returns lie in interval (-min_return, min_return).
       -2: If the vertical barrier is hit first and final returns lie in interval (lower barrier, -min_return].
        2: If the vertical barrier is hit first and final returns lie in interval [min_return, upper_barrier).
       42: Error. Both upper and lower horizontal barriers were hit at the same time.
    """
    if pt_multiplier <= 0:
        raise ValueError(
            f"Please ensure pt_multiplier is positive-valued. Provided pt_multiplier = {pt_multiplier}."
        )

    if sl_multiplier <= 0:
        raise ValueError(
            f"Please ensure sl_multiplier is positive-valued. Provided sl_multiplier = {sl_multiplier}."
        )

    return df.with_columns(
        create_barrier_label_plugin(
            price_expr=pl.col(price_col),
            volatility_expr=pl.col(volatility_col),
            pt_multiplier=pt_multiplier,
            sl_multiplier=sl_multiplier,
            vertical_barrier_window=vertical_barrier_window,
            min_return=min_return,
        )
        .over(partition_by=partition_by, order_by=order_by)
        .alias("triple_barrier_label")
    )
