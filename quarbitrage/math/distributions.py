"""
Module for common distributions
"""

import numpy as np


def get_ewma_weights(
    span: int, alpha: float = None, reverse: bool = True, unbiased: bool = True
) -> list[float]:
    """
    Compute ewma weights
    """
    if alpha is None:
        alpha = 2 / (span + 1)
    
    return create_geometric_sequence(
        x=1 - alpha, num_terms=span, reverse=reverse, normalize=unbiased
    )


def get_variance_bias_correction(weights: list[float]):
    """
    Compute bias correction for weighted variance
    """

    bias_correction = np.sum(weights) ** 2 / (
        np.sum(weights) ** 2 - np.sum(np.power(weights, 2))
    )

    return bias_correction


def create_geometric_sequence(
    x: float, num_terms: int, reverse: bool = False, normalize: bool = True
) -> list[float]:
    """
    Create geometric sequence

        [1, x, x^2,..., x^{num_terms-1}]
    """

    normalizer = 1 / get_geometric_sum(x=x, num_terms=num_terms) if normalize else 1

    geometric_seq = [
        normalizer * (x ** (num_terms - 1 - idx)) if reverse else normalizer * (x**idx)
        for idx in range(num_terms)
    ]

    return geometric_seq


def normalize_sequence(weights: list[float]) -> list[float]:
    """
    Normalize provided weights
    """
    total_weight = np.sum(weights)
    return [weight / total_weight for weight in weights]


def get_geometric_sum(x: float, num_terms: int) -> float:
    """
    Compute geometric sum for series

        1 + x + x^2 + ... + x^{num_terms-1}

    """
    geometric_sum = num_terms if x == 1 else (1 - x**num_terms) / (1 - x)
    return geometric_sum
