from abc import ABC, abstractmethod
from typing import Union

import polars as pl
from polars import DataFrame, LazyFrame

from quarbitrage.utils.file_handling import extract_csv_row_to_list, get_filetype
from quarbitrage.utils._exception_checks import check_if_columns_in_list


class Bars(ABC):
    """
    Abstract base class for BARS
    """

    def __init__(
        self,
        datetime_col: str = "DATETIME",
        price_col: str = "PRICE",
        volume_col: str = "VOLUME",
        lazy_evaluator: bool = True,
        scan_column_names: bool = True,
    ):
        """
        Constructor for Bars
        """
        self.datetime_col = datetime_col
        self.price_col = price_col
        self.volume_col = volume_col
        self.lazy_evaluator = lazy_evaluator
        self.scan_column_names = scan_column_names

    def set_lazy_evaluator(self, lazy_evaluator: bool):
        """
        Set .lazy_evaluator attribute to given boolean
        """
        assert isinstance(
            lazy_evaluator, bool
        ), "'lazy_evaluator' must be of type: bool"

        if lazy_evaluator == self.lazy_evaluator:
            print(
                f"Bars.lazy_evaluator attribute is already set to {self.lazy_evaluator}"
            )
        else:
            self.lazy_evaluator = lazy_evaluator

    def set_scan_column_names(self, scan_column_names: bool):
        """
        Set .scan_column_names attribute to given boolean
        """
        assert isinstance(
            scan_column_names, bool
        ), "'lazy_evaluator' must be of type: bool"

        if scan_column_names == self.scan_column_names:
            print(
                f".scan_column_names attribute is already set to {self.scan_column_names}"
            )
        else:
            self.scan_column_names = scan_column_names

    def _fetch_csv(
        self,
        csv_filepath: str,
        encoding: str = "utf8",
        **kwargs,
    ) -> Union[LazyFrame, DataFrame]:
        """
        Read .csv file into Polars LazyFrame or DataFrame
        """
        if self.scan_column_names:
            with open(csv_filepath, "r", encoding=encoding) as file:
                first_row = file.read_line()
                columns = extract_csv_row_to_list(first_row)
                self._check_for_required_columns(columns)

        return (
            pl.scan_csv(csv_filepath, encoding=encoding, **kwargs)
            if self.lazy_evaluator
            else pl.read_csv(csv_filepath, encoding=encoding, **kwargs)
        )

    def _fetch_excel(
        self, excel_filepath: str, **kwargs
    ) -> Union[LazyFrame, DataFrame]:
        """
        Read .excel file into Polars LazyFrame or DataFrame
        """
        pdf = pl.read_excel(excel_filepath, **kwargs)

        pdf = pdf.lazy() if self.lazy_evaluator else pdf

        if self.scan_column_names:
            columns = pdf.columns
            self._check_for_required_columns(columns)

        return pdf

    def _fetch_parquet(self, parquet_filepath, **kwargs) -> Union[LazyFrame, DataFrame]:
        """
        Read .parquet file into Polars LazyFrame or DataFrame
        """
        if self.scan_column_names:
            columns = list(pl.read_parquet_schema(parquet_filepath).keys())
            self._check_for_required_columns(columns)

        return (
            pl.scan_parquet(parquet_filepath, **kwargs)
            if self.lazy_evaluator
            else pl.read_parquet(parquet_filepath, **kwargs)
        )

    def _generate_dataframe(
        self,
        filepath_or_df: Union[str, LazyFrame, DataFrame],
    ) -> Union[LazyFrame, DataFrame]:
        """
        Generate Polars LazyFrame/Dataframe from file path
        """
        if isinstance(filepath_or_df, str):
            return self._generate_dataframe_from_filepath(filepath_or_df)

        elif isinstance(filepath_or_df, (LazyFrame, DataFrame)):
            return filepath_or_df
        else:
            raise TypeError(
                "'filepath_or_df' must be a string or Polars LazyFrame/DataFrame"
            )

    def _generate_dataframe_from_filepath(
        self, filepath: str
    ) -> Union[LazyFrame, DataFrame]:
        """
        Generate Polars LazyFrame/Dataframe from file path
        """

        assert isinstance(filepath, str), "File path must be a string."

        file_type = get_filetype(file_path=filepath)
        
        if file_type == "csv":
            df = self._fetch_csv(filepath)
        elif file_type == "excel":
            df = self._fetch_excel(filepath)
        elif file_type == "parquet":
            df = self._fetch_parquet(filepath)
        else:
            raise ValueError(f"File type '{file_type}' not recognized.")

        return df

    def _check_for_required_columns(self, columns: list[str]):
        """
        Check if required columns (timestamp, price, volume) are in provided columns list.
        """
        check_if_columns_in_list(
            [self.datetime_col, self.price_col, self.volume_col], columns
        )
