"""
Column Helper Module
"""

from typing import Union
from polars import DataFrame, LazyFrame

import polars as pl


def create_index(
    df: Union[LazyFrame, DataFrame], order_by: str = None, index_col: str = "index"
) -> Union[LazyFrame, DataFrame]:
    """
    Create index column
    """
    if order_by is not None:
        df = df.sort(by=order_by)

    return df.with_columns(pl.int_range(pl.len(), dtype=pl.UInt32).alias(index_col))


def transform_cols_to_index(
    df: Union[LazyFrame, DataFrame],
    columns: Union[str, list[str]],
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
    >>> df = transform_cols_to_index(df, columns = ["time_start", "time_end"])
    >>> df
    ----------------------------------------------------------------------
    | volume | time_start | time_end | time_start_index | time_end_index |
    ----------------------------------------------------------------------
    |   10   |   10:00    |   10:10  |        0         |        2       |
    |   20   |   10:05    |   10:10  |        1         |        2       |
    |   50   |   10:10    |   10:15  |        2         |        3       |
    ----------------------------------------------------------------------
    """

    if isinstance(columns, str):
        columns = [columns]

    values_df = get_all_columns_values(df, columns=columns)
    values_index_df = create_index(values_df, order_by="values")

    for column in columns:
        df = df.join(
            values_index_df.rename({"index": f"{column}_index"}),
            left_on=[column],
            right_on=["values"],
        )

    if return_index:
        return df, values_index_df

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
