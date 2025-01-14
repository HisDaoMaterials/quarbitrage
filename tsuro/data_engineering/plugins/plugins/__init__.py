from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from plugins._internal import __version__ as __version__

if TYPE_CHECKING:
    from plugins.typing import IntoExprColumn

LIB = Path(__file__).parent


def pig_latinnify(expr: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="pig_latinnify",
        is_elementwise=True,
    )


def create_barrier_label_plugin(
    price_expr: IntoExprColumn,
    volatility_expr: IntoExprColumn,
    pt_multiplier: float | None = 1.0,
    sl_multiplier: float | None = 1.0,
    vertical_barrier_window: int | None = None,
    min_return: float | None = None,
) -> pl.Expr:
    """
    Create Triple Barriers
    """

    if pt_multiplier is not None:
        upper_thresholds = volatility_expr * pt_multiplier
    else:
        upper_thresholds = None

    if sl_multiplier is not None:
        lower_thresholds = -volatility_expr * sl_multiplier
    else:
        lower_thresholds = None

    return register_plugin_function(
        args=[price_expr, lower_thresholds, upper_thresholds],
        plugin_path=LIB,
        function_name="create_triple_barrier_labels",
        is_elementwise=False,
        kwargs={
            "vertical_threshold": vertical_barrier_window,
            "min_return": min_return,
        },
    )
