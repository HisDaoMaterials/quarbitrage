import polars as pl
from polars import DataFrame, LazyFrame
from polars.datatypes.classes import DataTypeClass

from typing import Union, Optional
import tsuro.utils.error_handling as errors

def cast_strings_to_datetime(
        pdf: Union[DataFrame, LazyFrame],
        columns: Union[str, list[str]],
        format: Union[str, dict[str]],
        dtypes: Union[DataTypeClass, dict[DataTypeClass]] = pl.Datetime,
        strict: bool = False
    ) -> Union[DataFrame, LazyFrame]:
    """
    Cast polars dataframe string columns into datetime.
    
    Args:
        pdf: Polars DataFrame to cast
        columns: List of column names
        format: 
        dtypes: list of data types

    Returns:
        polars.DataFrame or polars.LazyFrame
    """
    if not isinstance(columns, (str,list)):
        raise ValueError(f"'columns' argument must be a string or a list. Received {columns} as argument.")

    if not isinstance(format, (str, dict)):
        raise ValueError(f"'format' argument must be a string or a dictionary. Received {format} as argument.")
    
    if isinstance(columns, str):
        columns = [columns]

    if isinstance(format, str):
        format = {column: format for column in columns}
    
    if isinstance(dtypes, DataTypeClass):
        print("dtypes true")
        dtypes = {column: pl.Datetime for column in columns}
    
    print(type(dtypes))
    # Check if len(columns) = len(format) = len(dtypes)
    errors.check_equal_size({"columns": columns, "format": format, "dtypes": dtypes})
    
    # Build casting expression
    casting_expression = [
        pl.col(column).str.strptime(
            dtype = dtypes[column], 
            format = format[column],
            strict = strict
        ).alias(column) 
        for column in columns
        ]

    pdf = pdf.with_columns(*casting_expression)
    
    return pdf
