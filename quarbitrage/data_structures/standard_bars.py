"""
StandardBars Class
"""

import polars as pl
from polars import LazyFrame, DataFrame
import polars.datatypes as T
from typing import Union

from quarbitrage.data_structures.bars import Bars
from quarbitrage.utils.file_handling import get_filetype


class StandardBars(Bars):
    """
    StandardBars Class
    """

    def __init__(
        self,
        datetime_col: str = "DATETIME",
        price_col: str = "PRICE",
        volume_col: str = "VOLUME",
        dollar_col: str = "DOLLAR",
        lazy_evaluator: bool = True,
        scan_column_names: bool = True,
    ):
        """
        Constructor for Standard Bars
        """
        super().__init__(
            datetime_col=datetime_col,
            price_col=price_col,
            volume_col=volume_col,
            lazy_evaluator=lazy_evaluator,
            scan_column_names=scan_column_names,
        )

        self.dollar_col = dollar_col

    def create_standard_bars(
        self,
        filepath_or_df: Union[str, LazyFrame, DataFrame],
        bar_size: float = 100000,
        bar_col: str = "volume",
        partition_by: Union[None, list[str]] = None,
        order_by: str = None,
        aggs: Union[str, list] = "ohlc",
    ) -> Union[LazyFrame, DataFrame]:
        """
        Create Standard bars

        Args:
            filepath_or_df:
            bar_size:
            bar_col:
            group_by:
            aggs: List of Polars aggregations to perform on bars
        """
        if aggs == "ohlc":
            aggs = [
                pl.first(self.price_col).alias("open_price"),
                pl.max(self.price_col).alias("high_price"),
                pl.min(self.price_col).alias("low_price"),
                pl.last(self.price_col).alias("close_price"),
                pl.first(self.datetime_col).alias("datetime_start"),
                pl.last(self.datetime_col).alias("datetime_end"),
            ]
        pdf = self._generate_dataframe(filepath_or_df=filepath_or_df)

        if order_by is not None:
            pdf = pdf.sort(by=order_by)

        pdf = (
            pdf.with_columns(
                pl.col(bar_col)
                .cum_sum()
                .over(partition_by=partition_by)
                .alias("cum_sum"),
            )
            .with_columns((pl.col("cum_sum") / bar_size).alias("bar_index"))
            .cast({"bar_index": T.Int32})
            .drop("cum_sum")
        )

        group_bars = (
            ["bar_index"] if partition_by is None else ["bar_index"] + partition_by
        )

        return pdf.group_by(group_bars).agg(*aggs)

    def create_volume_bars(
        self,
        filepath_or_df: Union[str, LazyFrame, DataFrame],
        volume_bar_size: float = 100000,
        partition_by: Union[None, list[str]] = None,
        order_by: str = None,
        aggs: Union[str, list] = "ohlc",
    ) -> Union[LazyFrame, DataFrame]:
        """
        Create Volume Bars
        """
        volume_bars = self.create_standard_bars(
            filepath_or_df=filepath_or_df,
            bar_size=volume_bar_size,
            bar_col=self.volume_col,
            partition_by=partition_by,
            order_by=order_by,
            aggs=aggs,
        )

        return volume_bars

    def create_dollar_bars(
        self,
        filepath_or_df: Union[str, LazyFrame, DataFrame],
        dollar_bar_size: float = 100000,
        partition_by: Union[None, list[str]] = None,
        order_by: str = None,
        aggs: Union[str, list] = "ohlc",
    ) -> Union[LazyFrame, DataFrame]:
        """
        Create Dollar Bars
        """
        dollar_bars = self.create_standard_bars(
            filepath_or_df=filepath_or_df,
            bar_size=dollar_bar_size,
            bar_col=self.dollar_col,
            partition_by=partition_by,
            order_by=order_by,
            aggs=aggs,
        )

        return dollar_bars
