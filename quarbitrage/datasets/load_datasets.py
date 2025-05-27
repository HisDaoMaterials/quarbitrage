from typing import Union
from importlib import resources

import polars as pl
from polars import DataFrame, LazyFrame


DATA_MODULE = "tsuro.datasets.data"

def load_cba_trades(
        lazy: bool = False
    ) -> Union[DataFrame, LazyFrame]:
    """
    Read CBA_trades.csv into dataframe with polars
    """
    data_path = resources.files(DATA_MODULE) / "cba_trades.csv"

    return load_csv(data_path, lazy = lazy)

def load_csv(
        file_path: str, 
        lazy: bool = False
    ) -> Union[DataFrame, LazyFrame]:
    """
    Load csv into Polars DataFrame/LazyFrame
    """

    pdf = pl.read_csv(file_path)
    pdf = pdf.lazy() if lazy else pdf
    
    return pdf

def load_excel(
        file_path: str,
        lazy: bool = False
    ) -> Union[DataFrame, LazyFrame]:
    """
    Load excel into Polars DataFrame/LazyFrame
    """
    pdf = pl.read_excel(file_path)
    pdf = pdf.lazy() if lazy else pdf
    
    return pdf
