"""
Column Helper Module
"""

from typing import Union, Optional
from polars import DataFrame, LazyFrame, Series

import polars as pl

import numpy as np


def create_dataframe_index(
    df: Union[LazyFrame, DataFrame],
    order_by: str = None,
    index_col: str = "index",
) -> Union[LazyFrame, DataFrame]:
    """
    Create index column
    """
    if order_by is not None:
        df = df.sort(by=order_by)

    return df.with_columns(pl.int_range(pl.len(), dtype=pl.UInt32).alias(index_col))


def create_series_index(
    series: Union[Series, DataFrame, list],
    series_alias: str = None,
    to_lazyframe: bool = True,
    index_col: str = "index",
) -> Union[LazyFrame, DataFrame]:
    """
    Create index column for series
    """
    assert isinstance(
        series, (pl.Series, DataFrame, list)
    ), "'series' argument must be Polars.Series, Polars.DataFrame or list."

    if isinstance(series, DataFrame):
        assert series.width == 1, "DataFrame must have only one column"
        series = series.sort(by=series.columns[0])

        if series_alias is not None:
            series = series.select(pl.col(series.columns[0]).alias(series_alias))

        series = series.lazy() if to_lazyframe else series
    else:
        if isinstance(series, list):
            series = pl.Series(values=series)

        series = series.sort()

        if series_alias is not None:
            series = series.alias(series_alias)

        series = pl.LazyFrame(series) if to_lazyframe else pl.DataFrame(series)

    return series.with_columns(pl.int_range(pl.len(), dtype=pl.UInt32).alias(index_col))


def transform_columns_to_index(
    df: Union[LazyFrame, DataFrame],
    index_columns: Union[str, list[str]],
    index: Union[Series, list] = None,
    return_index: bool = False,
) -> Union[LazyFrame, DataFrame]:
    """
    Create unified index label for all provided columns

    Examples:
    >>> df = pl.DataFrame({
        "volume": [10, 20, 50],
        "time_start": ["10:00", "10:05", "10:10"],
        "time_end": ["10:10", "10:10", "10:15"]
        }
    )
    >>> df1 = transform_columns_to_index(df, columns = ["time_start", "time_end"])
    >>> df1
    ----------------------------------------------------------------------
    | volume | time_start | time_end | time_start_index | time_end_index |
    ----------------------------------------------------------------------
    |   10   |   10:00    |   10:10  |        0         |        2       |
    |   20   |   10:05    |   10:10  |        1         |        2       |
    |   50   |   10:10    |   10:15  |        2         |        3       |
    ----------------------------------------------------------------------
    >>> index = pl.Series(values = ["10:00", "10:02", "10:05", "10:08", "10:10", "10:12", "10:15"])
    >>> df2 = transform_columns_to_index(df, columns = ["time_start", "time_end"], index = index)
    >>> df2
    ----------------------------------------------------------------------
    | volume | time_start | time_end | time_start_index | time_end_index |
    ----------------------------------------------------------------------
    |   10   |   10:00    |   10:10  |        0         |        4       |
    |   20   |   10:05    |   10:10  |        2         |        4       |
    |   50   |   10:10    |   10:15  |        4         |        6       |
    ----------------------------------------------------------------------
    """
    if isinstance(index, list):
        index = pl.Series(values=index)

    if isinstance(index_columns, str):
        index_columns = [index_columns]

    if index is None:
        index_values_df = get_all_columns_values(df, columns=index_columns)
        index = create_dataframe_index(index_values_df, order_by="values")
    else:
        index = create_series_index(
            index,
            series_alias="values",
            to_lazyframe=True if isinstance(df, pl.LazyFrame) else False,
        )

    for column in index_columns:
        df = df.join(
            index.rename({"index": f"{column}_index"}),
            left_on=[column],
            right_on=["values"],
        )

    if return_index:
        return df, index

    return df


def get_all_columns_values(
    df: Union[LazyFrame, DataFrame],
    columns: Union[str, list[str]],
    enforce_unique: bool = True,
) -> Union[LazyFrame, DataFrame]:
    """
    Grab all values from multiple columns and concatenate into a single column DataFrame
    """
    if isinstance(columns, str):
        columns = [columns]

    cols_values = pl.concat([df.select(pl.col(col).alias("values")) for col in columns])
    if enforce_unique:
        cols_values = cols_values.unique()

    return cols_values


def create_list_from_column(df: Union[LazyFrame, DataFrame], column: str) -> list:
    """
    Create list from column values
    """
    return df.select(column).to_series().sort().to_list()


def create_conjunctive_conditional(
    columns: Union[list[str], str], values: Union[float, str, list[any]]
) -> bool:
    """
    Create Polars conjuctive (AND) conditional based on provided columns and values
    """
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(values, list):
        assert len(columns) == len(values), "len(columns) must equal len(values)"
        conjunctive_conditional = pl.col(columns[0]) == values[0]
    else:
        conjunctive_conditional = pl.col(columns[0]) == values

    for idx in range(1, len(columns)):
        conjunctive_conditional = (
            conjunctive_conditional & (pl.col(columns[idx]) == values[idx])
            if isinstance(values, list)
            else conjunctive_conditional & (pl.col(columns[idx]) == values)
        )

    return conjunctive_conditional


def create_disjunctive_conditional(
    columns: Union[list[str], str], values: Union[float, str, list[any]]
) -> bool:
    """
    Create Polars disjunctive (OR) conditional based on provided columns and values
    """
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(values, list):
        assert len(columns) == len(values), "len(columns) must equal len(values)"
        disjunctive_conditional = pl.col(columns[0]) == values[0]
    else:
        disjunctive_conditional = pl.col(columns[0]) == values

    for idx in range(1, len(columns)):
        disjunctive_conditional = (
            disjunctive_conditional | (pl.col(columns[idx]) == values[idx])
            if isinstance(values, list)
            else disjunctive_conditional | (pl.col(columns[idx]) == values)
        )

    return disjunctive_conditional


def compute_column_sums(
    dataframe: DataFrame, columns: Optional[list] = None
) -> DataFrame:
    """
    Compute vertical sum of values across all columns
    """
    if columns is not None:
        dataframe = dataframe.select(columns)

    return dataframe.sum()


def compute_ndarray_sums(ndarray: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute sum across specified axis in a numpy.ndarray
    """
    return ndarray.sum(axis=axis)
