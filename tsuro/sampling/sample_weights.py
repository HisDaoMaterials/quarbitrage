"""
Sample Weights Module
"""

from typing import Union, Optional
from polars import DataFrame, LazyFrame, Series

import polars as pl

from tsuro.utils import (
    create_dataframe_index,
    transform_columns_to_index,
    create_list_from_column,
    create_conjunctive_conditional,
)


def create_overlap_matrix(
    bars_df: Union[DataFrame, LazyFrame],
    time_start_col: str,
    time_end_col: str,
    time_index: Union[Series, list] = None,
    index_col: Optional[str] = None,
    return_bars: bool = False,
    remove_no_overlaps: bool = True,
) -> Union[DataFrame, LazyFrame]:
    """
    Create overlap matrix

    Parameters
    ----------
        bars_df : Polars DataFrame / LazyFrame

        time_start_col : String

        time_end_col : String

        time_index : Optional[Series]

        index_col : Optional[String]

    Returns
    -------

    Examples:
    >>> df = pl.DataFrame({
        "bar_index": [1,2,3],
        "volume": [10, 20, 50],
        "time_start": ["10:00", "10:05", "10:10"],
        "time_end": ["10:10", "10:10", "10:15"]
        }
    )
    >>> index = pl.Series(values = ["10:00", "10:02", "10:05", "10:08", "10:10", "10:12", "10:15"])
    >>> overlap_matrix = create_overlap_matrix(df, time_start_col = "time_start", time_end_col = "time_end", time_index = index, index_col = "bar_index")
    >>> overlap_matrix
    ---------------------------------------------------------------------------------
    | index | time_start | time_end	| bar_1_overlap | bar_2_overlap	| bar_3_overlap |
    ---------------------------------------------------------------------------------
    |   0   |  "10:00"	 |  "10:02"	|       1	    |       0	    |       0       |
    |   1	|  "10:02"	 |  "10:05"	|       1	    |       0	    |       0       |
    |   2	|  "10:05"	 |  "10:08"	|       1	    |       1	    |       0       |
    |   3	|  "10:08"	 |  "10:10"	|       1	    |       1	    |       0       |
    |   4	|  "10:10"	 |  "10:12"	|       0	    |       0	    |       1       |
    |   5	|  "10:12"	 |  "10:15"	|       0	    |       0	    |       1       |
    ---------------------------------------------------------------------------------
    """
    if index_col is None:
        index_col = "index"
        bars_df = create_dataframe_index(bars_df, index_col=index_col)

    bars_df, time_index = transform_columns_to_index(
        bars_df,
        index_columns=[time_start_col, time_end_col],
        index=time_index,
        return_index=True,
    )

    bars_index = create_list_from_column(bars_df, index_col)

    time_index = setup_time_index(time_index)

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

    if remove_no_overlaps:
        remove_no_overlap_filter = create_conjunctive_conditional(
            columns=[f"bar_{idx}_overlap" for idx in bars_index], values=0
        )
        overlap_matrix = overlap_matrix.filter(remove_no_overlap_filter.not_())

    if return_bars:
        return overlap_matrix, bars_df
    else:
        return overlap_matrix


def setup_time_index(
    time_index: Union[DataFrame, LazyFrame],
    values_col: str = "values",
    index_col: str = "index",
) -> Union[DataFrame, LazyFrame]:
    """
    Setup time index dataframe for use with overlap matrix.
    """

    time_index = (
        time_index.with_columns(pl.col(values_col).shift(-1).alias("time_end"))
        .rename({values_col: "time_start"})
        .select(index_col, "time_start", "time_end")
    )

    return time_index
