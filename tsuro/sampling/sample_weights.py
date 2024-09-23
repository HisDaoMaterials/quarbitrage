"""
Sample Weights Module
"""

from typing import Union
from polars import DataFrame, LazyFrame

import polars as pl

from tsuro.utils import create_index, transform_cols_to_index


def create_overlap_matrix(
    bars_df: Union[DataFrame, LazyFrame],
    time_start_col: str,
    time_end_col: str,
    index_col: str = None,
) -> Union[DataFrame, LazyFrame]:
    """
    Create overlap matrix
    """
    if index_col is None:
        index_col = "index"
        bars_df = create_index(bars_df, index_col=index_col)

    bars_df, time_index = transform_cols_to_index(
        bars_df, columns=[time_start_col, time_end_col], return_index=True
    )

    bars_index = bars_df.select(index_col).to_series().to_list()

    time_index = (
        time_index.with_columns(pl.col("values").shift(-1).alias("time_end"))
        .rename({"values": "time_start"})
        .select("index", "time_start", "time_end")
    )

    overlap_matrix = time_index.with_columns(
        pl.when(
            (
                pl.col("index")
                >= bars_df.filter(pl.col(index_col) == idx).select(
                    f"{time_start_col}_index"
                )
            )
            & (
                pl.col("index")
                < bars_df.filter(pl.col(index_col) == idx).select(
                    f"{time_end_col}_index"
                )
            )
        )
        .then(1)
        .otherwise(0)
        .alias(f"bar_{idx}_overlap")
        for idx in bars_index
    ).filter(pl.col("index") < time_index.shape[0] - 1)

    return overlap_matrix, bars_df
