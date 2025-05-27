import polars as pl
import pandas as pd
import polars.datatypes as T
import numpy as np

from datetime import datetime, time
import sys

sys.path.insert(0, "C:\\Users\\rockh\\Repositories\\Tsuro\\")

from quarbitrage.datasets import load_cba_trades
from quarbitrage.preprocessing import cast_strings_to_datetime

import polars.datatypes as T

test_df = pl.DataFrame(
    {
        "hour": [12, 10, 13, 10, 12, 10, 11, 11],
        "minute": [20, 5, 15, 15, 12, 10, 15, 25],
        "price": [90.1, 90.5, 95.3, 93.4, 96.7, 88.3, 85.4, 87.6],
        "volatility": [0.02, 0.05, 0.07, 0.015, 0.03, 0.04, 0.02, 0.03],
    }
).with_columns(pl.time(pl.col("hour"), pl.col("minute")).alias("time"))


from quarbitrage.data_engineering import create_triple_barrier_labels

df = create_triple_barrier_labels(
    test_df,
    price_col="price",
    volatility_col="volatility",
    order_by="time",
    vertical_barrier_window=3,
    min_return=0.02,
)

print(df)
