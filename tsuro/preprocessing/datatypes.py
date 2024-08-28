import polars as pl
from polars import DataFrame, LazyFrame
from polars.datatypes.classes import DataTypeClass

from typing import Union, Optional
import tsuro.utils.exception_checks as ec


def cast_strings_to_datetime(
    pdf: Union[DataFrame, LazyFrame],
    columns: Union[str, list[str]],
    dtypes: Union[DataTypeClass, dict[DataTypeClass]] = pl.Datetime,
    datetime_format: Union[str, dict[str]] = r"%m/%d/%Y %H:%M",
    strict: bool = True,
) -> Union[DataFrame, LazyFrame]:
    """
    Cast Polars DataFrame string columns into datetime datatypes.

    Args:
        pdf: Polars DataFrame to cast
        columns: List of column names
        datetime_format: String or Dictionary mapping column names to date/datetime format for parsing
        dtypes: DataTypeClass or Dictionary mapping column names to DataTypeClass.
        strict:
    
    Returns:
        polars.DataFrame or polars.LazyFrame
    """
    if not isinstance(columns, (str, list)):
        raise TypeError(
            f"'columns' argument must be a string or a list. Received {type(columns)} object instead."
        )

    if not isinstance(datetime_format, (str, dict)):
        raise TypeError(
            f"'format' argument must be a string or a dictionary. Received {type(datetime_format)} object instead."
        )

    if isinstance(columns, str):
        columns = [columns]

    if isinstance(datetime_format, str):
        datetime_format = {column: datetime_format for column in columns}

    if isinstance(dtypes, DataTypeClass):
        print("dtypes true")
        dtypes = {column: dtypes for column in columns}

    ec.check_equal_length(
        {"columns": columns, "datetime_format": datetime_format, "dtypes": dtypes}
    )

    # Build casting expression
    casting_expression = [
        pl.col(column)
        .str.strptime(
            dtype=dtypes[column], format=datetime_format[column], strict=strict
        )
        .alias(column)
        for column in columns
    ]

    pdf = pdf.with_columns(*casting_expression)

    return pdf
