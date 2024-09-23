"""
Testing Module
"""

from typing import Union
from polars.testing import assert_frame_equal, assert_frame_not_equal
from polars import DataFrame, LazyFrame


def assert_polars_frame_equal(
    left_df: Union[DataFrame, LazyFrame],
    right_df: Union[DataFrame, LazyFrame],
    check_row_order: bool = True,
    check_column_order: bool = False,
    check_dtypes: bool = True,
    check_exact: bool = False,
    rel_tol: float = 1e-05,
    abs_tol: float = 1e-08,
    categorical_as_str: bool = False,
) -> None:
    """
    Determine whether left Polars Frame = right Polars Frame up to some tolerance factor specified with rel_tol / abs_tol
    """
    assert_frame_equal(
        left=left_df,
        right=right_df,
        check_row_order=check_row_order,
        check_column_order=check_column_order,
        check_dtypes=check_dtypes,
        check_exact=check_exact,
        rtol=rel_tol,
        atol=abs_tol,
        categorical_as_str=categorical_as_str,
    )
