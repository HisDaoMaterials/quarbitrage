"""
Bootstrapping functions for sequential bootstrapping and overlap matrix creation.
"""

from typing import Union, Optional
from polars import DataFrame, LazyFrame, Series

import polars as pl
import numpy as np

from tsuro.utils.random import random_discrete_draw
from tsuro.utils import (
    create_dataframe_index,
    transform_columns_to_index,
    create_list_from_column,
    compute_ndarray_sums,
)


def _update_index_indicators(index_indicator_array: np.array, index: int) -> np.array:
    """
    Update index array
    """
    index_indicator_array[index] = 1

    return index_indicator_array


def _update_sample_indices(sample_indices: np.array, idx: int, result: int) -> np.array:
    """
    Update sample array
    """
    sample_indices[idx] = int(result)
    return sample_indices


def _compute_uniqueness(
    overlap_matrix: np.ndarray, index_indicator_array: np.array
) -> DataFrame:
    """
    Compute Uniqueness
    """
    weights = 1 + np.dot(overlap_matrix, index_indicator_array)

    weights = weights.reshape(-1, 1)
    return overlap_matrix / weights


def _compute_average_uniqueness(
    uniqueness_matrix: np.ndarray, overlap_matrix_column_sums: np.ndarray
) -> np.ndarray:
    """
    Compute Average Uniqueness
    """
    return compute_ndarray_sums(uniqueness_matrix) / overlap_matrix_column_sums


def _compute_probabilities(avg_uniqueness_vector: np.ndarray) -> np.array:
    """
    Compute Probabilities
    """
    probs = avg_uniqueness_vector / avg_uniqueness_vector.sum(axis=1)

    return probs.reshape(-1)


def create_sequential_bootstrap_indices(
    overlap_matrix: np.ndarray,
    overlap_matrix_column_sums: np.ndarray,
    sample_index_pool: Union[np.array, list],
    max_samples: int = -1,
    random_state=np.random.RandomState(),
) -> np.array:
    """
    Create sample indices using sequential bootstrapping
    """
    n_indices = overlap_matrix.shape[1]

    if max_samples < 0:
        max_samples = n_indices + 1 + max_samples

    # Initialization
    sample_indices = np.zeros(max_samples)
    index_indicator_array = np.zeros(n_indices)

    # Initial Draw
    random_draw = random_discrete_draw(sample_index_pool, random_state=random_state)

    # Update Index Indicator Array and Sample Indices
    sample_indices = _update_sample_indices(sample_indices, 0, random_draw)
    index_indicator_array = _update_index_indicators(index_indicator_array, random_draw)

    n_samples_drawn = 1
    while n_samples_drawn < max_samples:
        # Compute Uniqueness and Update Probability Distribution
        uniqueness_matrix = _compute_uniqueness(overlap_matrix, index_indicator_array)

        avg_uniqueness_vector = _compute_average_uniqueness(
            uniqueness_matrix, overlap_matrix_column_sums
        )

        probs = _compute_probabilities(avg_uniqueness_vector)

        # Random draw according to Probability Distribution
        random_draw = random_discrete_draw(
            sample_index_pool, probs, random_state=random_state
        )

        # Update Index Indicator Array
        index_indicator_array = _update_index_indicators(
            index_indicator_array, random_draw
        )
        sample_indices = _update_sample_indices(
            sample_indices, n_samples_drawn, random_draw
        )

        # Increase Number of Samples Drawn
        n_samples_drawn = n_samples_drawn + 1

    return sample_indices.astype(int)


def create_overlap_matrix(
    bars_df: Union[DataFrame, LazyFrame],
    time_start_col: Optional[Union[str, int]] = None,
    time_end_col: Optional[Union[str, int]] = None,
    time_index: Union[Series, list] = None,
    index_col: Optional[str] = None,
    return_bars: bool = False,
    remove_no_overlaps: bool = True,
) -> Union[DataFrame, LazyFrame]:
    """
    Create overlap indicator matrix between all time intervals and bar indices start / end timestamps.

    Parameters
    ----------
    bars_df : Polars DataFrame / LazyFrame
        DataFrame containing bars information

    time_start_col : Optional[String, int]
        If string, name of start time column. If integer, index of start time column

    time_end_col : Optional[String, int]
        If string, name of end time column. If integer, index of end time column

    time_index : Optional[Series]
        Time index to use for overlap matrix

    index_col : Optional[String]
        Name of index column

    return_bars : bool
        Return bars DataFrame along with overlap matrix

    remove_no_overlaps : bool
        Remove time intervals with no bar overlaps

    Returns
    -------
    overlap_matrix : Polars DataFrame / LazyFrame
        DataFrame containing overlap indicators between time intervals and bars

    Examples:
    >>> df = pl.DataFrame({
        "bar_index": [0,1,2],
        "time_start": ["10:00", "10:05", "10:10"],
        "time_end": ["10:10", "10:10", "10:15"]
        }
    )
    >>> index = pl.Series(values = ["10:00", "10:02", "10:05", "10:08", "10:10", "10:12", "10:15"])
    >>> overlap_matrix = create_overlap_matrix(
        df,
        time_start_col = "time_start",
        time_end_col = "time_end",
        time_index = index,
        index_col = "bar_index"
    )
    >>> overlap_matrix
    ---------------------------------------------------------------------------------
    | index | time_start | time_end	| bar_0_overlap | bar_1_overlap	| bar_2_overlap |
    ---------------------------------------------------------------------------------
    |   0   |  "10:00"	 |  "10:02"	|       1	    |       0	    |       0       |
    |   1	|  "10:02"	 |  "10:05"	|       1	    |       0	    |       0       |
    |   2	|  "10:05"	 |  "10:08"	|       1	    |       1	    |       0       |
    |   3	|  "10:08"	 |  "10:10"	|       1	    |       1	    |       0       |
    |   4	|  "10:10"	 |  "10:12"	|       0	    |       0	    |       1       |
    |   5	|  "10:12"	 |  "10:15"	|       0	    |       0	    |       1       |
    ---------------------------------------------------------------------------------
    """
    print("Creating Overlap Matrix...")
    if time_start_col is None:
        time_start_col = 0 if index_col is None else 1
    if time_end_col is None:
        time_end_col = 1 if index_col is None else 2

    if isinstance(time_start_col, int):
        time_start_col = bars_df.columns[time_start_col]
    if isinstance(time_end_col, int):
        time_end_col = bars_df.columns[time_end_col]

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
        overlap_matrix = (
            overlap_matrix.with_columns(
                pl.sum_horizontal(
                    pl.all().exclude("index", "time_start", "time_end")
                ).alias("overlap_sum")
            )
            .filter(pl.col("overlap_sum") > 0)
            .drop("overlap_sum")
        )

    if return_bars:
        return overlap_matrix, bars_df
    else:
        return overlap_matrix


def create_concurrent_overlap(
    overlap_matrix: Union[DataFrame, LazyFrame],
    bar_col_start_idx: int = 3,
    drop_bar_cols: bool = True,
    concurrent_alias: str = "concurrency_count",
) -> DataFrame:
    """
    Compute concurrent overlap across all bars for each time interval.

    Examples:
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
    >>> concurrent_overlap = create_concurrent_overlap(overlap_matrix)
    >>> concurrent_overlap
    ---------------------------------------------------
    | index | time_start | time_end	|concurrency_count|
    ---------------------------------------------------
    |   0   |  "10:00"	 |  "10:02"	|        1        |
    |   1	|  "10:02"	 |  "10:05"	|        1	      |
    |   2	|  "10:05"	 |  "10:08"	|        2	      |
    |   3	|  "10:08"	 |  "10:10"	|        2	      |
    |   4	|  "10:10"	 |  "10:12"	|        1	      |
    |   5	|  "10:12"	 |  "10:15"	|        1	      |
    ---------------------------------------------------
    """
    bar_cols = overlap_matrix.columns[bar_col_start_idx::]

    concurrent_vector = overlap_matrix.with_columns(
        pl.sum_horizontal(bar_cols).alias(concurrent_alias)
    )

    if drop_bar_cols:
        concurrent_vector = concurrent_vector.drop(bar_cols)

    return concurrent_vector


def create_bar_sizes(
    overlap_matrix: Union[DataFrame, LazyFrame], bar_col_start_idx: int = 3
) -> DataFrame:
    """
    Compute sizes for each bar column
    """
    bar_cols = overlap_matrix.columns[bar_col_start_idx::]

    return overlap_matrix.select(bar_cols).sum()


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
