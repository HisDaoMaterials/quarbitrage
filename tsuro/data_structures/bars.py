from abc import ABC, abstractmethod
from typing import Union

import polars as pl
from polars import DataFrame, LazyFrame

from tsuro.sql import DatabaseClient
from tsuro.utils.file_handling import extract_csv_row_to_list
from tsuro.utils.exception_checks import check_if_columns_in_list


class Bars(ABC):
    """
    Abstract base class for BARS
    """

    def __init__(
        self,
        database_client: DatabaseClient = None,
        datetime_col: str = "DATETIME",
        price_col: str = "PRICE",
        volume_col: str = "VOLUME",
    ):
        """
        Constructor for Bars
        """
        self.database_client = database_client
        self.datetime_col = datetime_col
        self.price_col = price_col
        self.volume_col = volume_col
        self.lazy_evaluator: bool = True
        self.pdf: Union[LazyFrame, DataFrame] = None

    def set_lazy_evaluator(self, lazy_evaluator: bool):
        """
        Set lazy_evaluator attribute to given boolean
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
            if self.pdf is not None:
                self._switch_pdf_frame()

    def set_database_client(self, database_client: DatabaseClient):
        """
        Set database_client attribute to given DatabaseClient
        """
        assert isinstance(
            database_client, DatabaseClient
        ), "'database_client' must be of type: DatabaseClient"
        self.database_client = database_client

    def fetch_csv(
        self,
        csv_filepath: str,
        scan_column_names: bool = True,
        encoding: str = "utf8",
        **kwargs,
    ) -> Union[LazyFrame, DataFrame]:
        """
        Read .csv file into Polars LazyFrame or DataFrame
        """
        if scan_column_names:
            with open(csv_filepath, "r", encoding=encoding) as file:
                first_row = file.read_line()
                columns = extract_csv_row_to_list(first_row)
                self._check_for_required_columns(columns)

        self.pdf = (
            pl.scan_csv(csv_filepath, encoding=encoding, **kwargs)
            if self.lazy_evaluator
            else pl.read_csv(csv_filepath, encoding=encoding, **kwargs)
        )

    def fetch_excel(
        self, excel_filepath: str, scan_column_names: bool = True, **kwargs
    ) -> Union[LazyFrame, DataFrame]:
        """
        Read .excel file into Polars LazyFrame or DataFrame
        """
        pdf = pl.read_excel(excel_filepath, **kwargs)

        pdf = pdf.lazy() if self.lazy_evaluator else pdf

        if scan_column_names:
            columns = pdf.columns
            self._check_for_required_columns(columns)

        self.pdf = pdf

    def fetch_parquet(
        self, parquet_filepath, scan_column_names: bool = True, **kwargs
    ) -> Union[LazyFrame, DataFrame]:
        """
        Read .parquet file into Polars LazyFrame or DataFrame
        """
        if scan_column_names:
            columns = list(pl.read_parquet_schema(parquet_filepath).keys())
            self._check_for_required_columns(columns)

        self.pdf = (
            pl.scan_parquet(parquet_filepath, **kwargs)
            if self.lazy_evaluator
            else pl.read_parquet(parquet_filepath, **kwargs)
        )

    def fetch_table(self, *args, **kwargs) -> Union[LazyFrame, DataFrame]:
        """
        Read table into Polars LazyFrame / DataFrame using DatabaseClient.read_table()

        Parameters:
            *args: Positional arguments relevant for self.database_client.read_table()
            **kwargs: Keyword arguments relevant for self.database_client.read_table()
        """
        if self.database_client is None:
            print(
                "No .database_client attribute provided. Please provide DatabaseClient via .set_database_client() method"
            )
        else:
            self.pdf = self.database_client.read_table(
                *args, **kwargs, lazy_evaluator=self.lazy_evaluator
            )

    def fetch_query(self, *args, **kwargs) -> Union[LazyFrame, DataFrame]:
        """
        Read SQL query into polars LazyFrame or DataFrame using DatabaseClient.read_query()

        Parameters:
            *args: Positional arguments for self.database_client.read_query()
            **kwargs: Keyword arguments for self.database_client.read_query()
        """
        if self.database_client is None:
            print(
                "No .database_client attribute provided. Please provide DatabaseClient via .set_database_client() method"
            )
        else:
            self.pdf = self.database_client.read_query(
                *args, **kwargs, lazy_evaluator=self.lazy_evaluator
            )

    def _switch_pdf_frame(self):
        """
        Convert Polars DataFrame into LazyFrame or vice-versa
        """
        if self.lazy_evaluator:
            self.pdf = self.pdf.lazy()  # Convert to LazyFrame
        else:
            self.pdf = self.pdf.collect()  # Convert to DataFrame

    def _check_for_required_columns(self, columns: list[str]):
        """
        Check if required columns (timestamp, price, volume) are in provided columns list.
        """
        check_if_columns_in_list(
            [self.datetime_col, self.price_col, self.volume_col], columns
        )

    @abstractmethod
    def create_bars(self):
        """Create BARS"""
