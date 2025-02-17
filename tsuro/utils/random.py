"""
Helper Module for Random Sampling
"""

from polars import DataFrame

from typing import Union, Optional

import numpy as np


def random_discrete_draw(
    index_pool: int,
    prob_dist: Optional[np.array] = None,
    random_state: np.random.RandomState = np.random.RandomState(),
) -> int:
    """
    Random Draw from elements using provided probability distribution
    """
    return random_state.choice(index_pool, p=prob_dist)


def horizontal_normalization(
    df: DataFrame, row: Optional[int] = None
) -> Union[np.array, DataFrame]:
    """
    Normalize dataframe horizontally
    """

    df_normed = df / df.sum_horizontal()

    if row is not None:
        df_normed = df_normed.row(row)

    return df_normed
