"""
Compute distributions

TODO: Generalize ewma to geometric distribution
"""


def get_ewma_weights(
    span: int, alpha: float = None, reverse: bool = True, unbiased: bool = True
):
    """
    Compute ewma weights
    """
    if alpha is None:
        alpha = 2 / (span + 1)

    bias_correction = (1 - alpha) / ((1 - alpha) ** (span - 1)) if unbiased else 1

    if reverse:
        ewma_weights = [
            bias_correction * (1 - alpha) ** (span - 1 - idx) for idx in range(span)
        ]
    else:
        ewma_weights = [bias_correction * (1 - alpha) ** idx for idx in range(span)]

    return ewma_weights
