import pytest

import polars as pl
from quarbitrage.utils._testing import assert_polars_frame_equal
from quarbitrage.data_engineering.rolling_moments import ewmstd


TEST_EWM_DF = pl.DataFrame({
    "class": [0,1,0,1,0,1],
    "hour": [12,10,13,10,12,10],
    "minute": [20,5,15,15,12,10],
    "volume": [150,16,2,5,25,40]
    }
).with_columns(
    pl.time(pl.col("hour"), pl.col("minute")).alias("time")
)

def test_ewmstd():
    """
    Test data_engineering.rolling_moments.ewmstd() function.
    """
    volume_ewmstd_span2_series = pl.Series(
        "volume_ewmstd_span2", []
    )
    volume_ewmstd_span3_series = pl.Series(
        "volume_ewmstd_span3", [
            "Volume 2:" 86.312968732564, 
            "Volume 5:" 20.209615815956806, 
            ]
    )
    
    expected_df = TEST_EWM_DF.insert_column(
        4, volume_ewmstd_span2_series
    ).insert_column(
        5, volume_ewmstd_span3_series
    )

    calculated_df = ewmstd(
        TEST_EWM_DF,
        ewmstd_cols=["volume"],
        spans=[2, 3],
        partition_by="class",
        order_by="time",
    )

    assert_polars_frame_equal(calculated_df, expected_df)
