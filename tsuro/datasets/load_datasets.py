from typing import Union
from importlib import resources

import polars as pl
from polars import DataFrame as PolarsDataFrame
from polars import LazyFrame as PolarsLazyFrame


DATA_MODULE = "tsuro.datasets.data"

def load_cba_trades(
        lazy: bool = False
    ) -> Union[PolarsDataFrame, PolarsLazyFrame]:
    """
    Read CBA_trades.csv into dataframe with polars
    """
    data_path = resources.files(DATA_MODULE) / "CBA_trades.csv"

    return load_csv(data_path, lazy = lazy)

def load_csv(
        file_path: str, 
        lazy: bool = False
    ) -> Union[PolarsDataFrame, PolarsLazyFrame]:
    """
    Load csv into Polars DataFrame/LazyFrame
    """
    
    pdf = pl.read_csv(file_path)
    pdf = pdf.lazy() if lazy else pdf
    
    return pdf

def load_excel(
        file_path: str,
        lazy: bool = False
    ) -> Union[PolarsDataFrame, PolarsLazyFrame]:
    """
    Load excel into Polars DataFrame/LazyFrame
    """
    pdf = pl.read_excel(file_path)
    pdf = pdf.lazy() if lazy else pdf
    
    return pdf
