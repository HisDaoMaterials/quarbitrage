from typing import Optional, Union
from abc import ABCMeta, abstractmethod
import numpy as np

from polars import DataFrame, Series
from sklearn.ensemble._bagging import BaseBagging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import validate_data, check_is_fitted, has_fit_parameter
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement

from tsuro.sampling.bootstrapping import create_sequential_bootstrap_indices, create_overlap_matrix


def _generate_indices(
    n_population: int,
    n_samples: int,
    bootstrap: bool = False,
    random_state: np.random.RandomState = np.random.RandomState(),
) -> np.array:
    """
    Generate random indices
    """

    if bootstrap:
        indices = random_state.randint(0, n_population, n_samples)
    else:
        indices = sample_without_replacement(
            n_population, n_samples, random_state=random_state
        )
    return indices


def _generate_sequential_bagging_indices(
    overlap_matrix: np.ndarray,
    overlap_matrix_column_sums: np.ndarray,
    sample_index_pool: np.array,
    max_samples: int,
    n_features: int,
    max_features: int,
    bootstrap_features: bool = False,
    random_state=np.random.RandomState(),
) -> tuple:
    """
    Generate random indices for sequential bagging
    """
    # Validate random state
    random_state = check_random_state(random_state)

    # Draw sample & feature indices
    sample_indices = create_sequential_bootstrap_indices(
        overlap_matrix=overlap_matrix,
        overlap_matrix_column_sums=overlap_matrix_column_sums,
        sample_index_pool=sample_index_pool,
        max_samples=max_samples,
    )

    feature_indices = _generate_indices(
        n_features,
        max_features,
        bootstrap=bootstrap_features,
        random_state=random_state,
    )

    return feature_indices, sample_indices


def _parallel_build_estimators(
    n_estimators: int,
    ensemble,
    X,
    y,
    overlap_matrix: np.ndarray,
    overlap_matrix_column_sums: np.ndarray,
    sample_weight: Optional[np.array],
    seeds,
    total_n_estimators: int,
    verbose: int,
) -> list:
    """
    Private function used to build a batch of estimators within a job.
    """
    # Retrieve Settings
    n_samples, n_features = X.shape
    n_timestamps, n_indices = overlap_matrix.shape
    max_features = ensemble._max_features
    max_samples = ensemble._max_samples
    bootstrap_features = ensemble.bootstrap_features
    sample_index_pool = ensemble.sample_index_pool
    support_sample_weight = has_fit_parameter(ensemble.base_estimator_, "sample_weight")

    # Build Estimators
    estimators = []
    estimators_features = []
    estimators_sample_indices = []

    assert (
        n_samples == n_indices
    ), "Number of samples (X.shape[1]) and overlap matrix indices (overlap_matrix.shape[1]) must be equal."

    if not support_sample_weight and sample_weight is not None:
        raise ValueError(
            "The base estimator doesn't support sample weight, but sample_weight has been passed to the fit method."
        )

    for i in range(n_estimators):
        if verbose > 1:
            print(
                f"Building estimator {i+1} of {n_estimators} for this parallel run (total {total_n_estimators})..."
            )

        random_state = np.random.RandomState(seeds[i])
        estimator = ensemble._make_estimator(append=False, random_state=random_state)

        # Draw Features & Samples
        feature_indices, sample_indices = _generate_sequential_bagging_indices(
            overlap_matrix=overlap_matrix,
            overlap_matrix_column_sums=overlap_matrix_column_sums,
            sample_index_pool=sample_index_pool,
            max_samples=max_samples,
            n_features=n_features,
            max_features=max_features,
            bootstrap_features=bootstrap_features,
            random_state=random_state,
        )

        if support_sample_weight:
            if sample_weight is None:
                fit_sample_weight = np.ones(n_samples)
            else:
                fit_sample_weight = sample_weight.copy()

            sample_counts = np.bincount(sample_indices, minlength=n_samples)
            fit_sample_weight *= sample_counts

            estimator.fit(X[:, feature_indices], y, sample_weight=fit_sample_weight)
        else:
            estimator.fit((X[sample_indices])[:, feature_indices], y[sample_indices])

        estimators.append(estimator)
        estimators_features.append(feature_indices)
        estimators_sample_indices.append(sample_indices)

    return estimators, estimators_features, estimators_sample_indices


class SequentialBootstrapBaseBagging(BaseBagging, metaclass=ABCMeta):
    """
    Base class for Sequential Bootstrap Classifier and Regressor. This extends the BaseBagging class from sklearn.

    Warning: This class should not be used directly. Use derived classes instead.
    """

    @abstractmethod
    def __init__(
        self,
        bars: DataFrame,
        time_index: Optional[Series] = None,
        overlap_matrix: Optional[np.ndarray] = None,
        estimator=None,
        n_estimators: int = 10,
        max_samples: Union[float, int] = 1.0,
        max_features: Union[float, int] = 1.0,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        warm_start: bool = False,
        n_jobs: Optional[int]=None,
        random_state=None,
        verbose: int =0,
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

        self.bars = bars
        self.overlap_matrix = overlap_matrix if overlap_matrix is not None else create_overlap_matrix(bars)

    def fit(
            self,
            X: DataFrame,
            y: Union[Series, DataFrame],
            sample_weight = None
    ):
        """
        Build a Sequential Bootstrap ensemble of estimators from the training set (X, y).

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note: This parameter is only supported for estimators that support sample weights.

        fit_params: dict
            Additional fit parameters.

        Returns
        -------
        self : SequentialBootstrapBaseBagging
            The fitted estimator.
        """
        return self._fit(X,y, self.max_samples, sample_weight=sample_weight)
    
    def _fit(
            self,
            X: DataFrame,
            y: Union[Series, DataFrame],
            sample_weight = None
    ):
        # Validate input data
        X, y = validate_data(X, y, multi_output=True, accept_sparse=False)
        check_classification_targets(y)
        self._le = self._validate_y(y)

        # Validate sample weights
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError("Sample weights must be of the same length as the input data.")
        self.sample_weight = sample_weight

        # Set up the ensemble
        self._setup_estimators()
        self._seeds = np.random.randint(MAX_INT, size=self.n_estimators)

        # Build the ensemble
        self.estimators_, self.estimators_features_, self.estimators_sample_indices_ = _parallel_build_estimators(
            n_estimators=self.n_estimators,
            ensemble=self,
            X=X,
            y=y,
            overlap_matrix=self.overlap_matrix,
            overlap_matrix_column_sums=self.overlap_matrix.sum(axis=0),
            sample_weight=self.sample_weight,
            seeds=self._seeds,
            total_n_estimators=self.n_estimators,
            verbose=self.verbose,
        )

        # Compute OOB Score
        if self.oob_score:
            self._oob_score = self._compute_oob_score(X, y)

        return self
        
