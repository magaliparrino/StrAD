# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, version 3.
#
# --- ADAPTATION NOTICE ---
# This function is adapted from [pyod] by [yzhao062] 
# Original source: [https://github.com/yzhao062/pyod]
#



import numbers
from pysad.core.base_model import BaseModel
from utils.utility import get_optimal_n_bins
from sklearn.utils import check_array
from collections import deque
from typing import List, Tuple, Optional
import numpy as np


class LODA(BaseModel):
    """The LODA model :cite:`pevny2016loda` The implemnetation is adapted to the steraming framework from the `PyOD framework <https://pyod.readthedocs.io/en/latest/_modules/pyod/models/loda.html#LODA>`_.

        Args:
            num_bins (int): The number of bins of the histogram.
            num_random_cuts (int): The number of random cuts.
    """

    def __init__(self, num_bins: int = 'auto', num_random_cuts=100):
        self.n_bins = num_bins
        self.num_random_cuts = num_random_cuts
        self.weights = np.ones(num_random_cuts, dtype=float) / num_random_cuts
        self.histograms_ = []


    def fit(self, X_train: np.ndarray,y=None) -> 'LODA':
        """Fit detector. y is ignored in unsupervised methods.

        args:
            X (np.ndarray): The input samples. Shape (n_samples, n_features).

        returns:
            self (LODA): The fitted estimator.
        """
        X_train = check_array(X_train)
        pred_scores = np.zeros([X_train.shape[0], 1])
        n_components = X_train.shape[1]
        n_nonzero_components = np.sqrt(n_components)
        n_zero_components = n_components - int(n_nonzero_components)

        self.projections_ = np.random.randn(self.num_random_cuts, n_components)

        # If set to auto: determine optimal n_bins using Birge Rozenblac method
        if isinstance(self.n_bins, str) and self.n_bins.lower() == "auto":

            self.limits_ = []
            self.n_bins_ = []  # only used when n_bins is determined by method "auto"

            for i in range(self.num_random_cuts):
                rands = np.random.permutation(n_components)[:n_zero_components]
                self.projections_[i, rands] = 0.
                projected_data = self.projections_[i, :].dot(X_train.T)

                n_bins = get_optimal_n_bins(projected_data)
                self.n_bins_.append(n_bins)

                histogram, limits = np.histogram(
                    projected_data, bins=n_bins, density=False)
                histogram = histogram.astype(np.float64)
                histogram += 1e-12
                histogram /= np.sum(histogram)

                self.histograms_.append(histogram)
                self.limits_.append(limits)

                # calculate the scores for the training samples
                inds = np.searchsorted(limits[:n_bins - 1],
                                       projected_data, side='left')
                pred_scores[:, 0] += -self.weights[i] * np.log(
                    histogram[inds])

        elif isinstance(self.n_bins, numbers.Integral):

            self.histograms_ = np.zeros((self.num_random_cuts, self.n_bins))
            self.limits_ = np.zeros((self.num_random_cuts, self.n_bins + 1))

            for i in range(self.num_random_cuts):
                rands = np.random.permutation(n_components)[:n_zero_components]
                self.projections_[i, rands] = 0.
                projected_data = self.projections_[i, :].dot(X_train.T)
                self.histograms_[i, :], self.limits_[i, :] = np.histogram(
                    projected_data, bins=self.n_bins, density=False)
                self.histograms_[i, :] += 1e-12
                self.histograms_[i, :] /= np.sum(self.histograms_[i, :])

                # calculate the scores for the training samples
                inds = np.searchsorted(self.limits_[i, :self.n_bins - 1],
                                       projected_data, side='left')
                pred_scores[:, 0] += -self.weights[i] * np.log(
                    self.histograms_[i, inds])

        else:
            raise ValueError("n_bins must be an int or \'auto\', "
                             "got: %f" % self.n_bins)

        return self


    def fit_partial(self, X, y=None):
        """Fits the model to next instance.

        Args:
            X : numpy array of shape (n_samples, n_features)
            The training input samples.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: Returns the self.
        """
        X = X.reshape(1, -1) if X.ndim == 1 else X
        X = check_array(X)
        pred_scores = np.zeros([X.shape[0], 1])

        if isinstance(self.n_bins, str) and self.n_bins.lower() == "auto":

            for i in range(self.num_random_cuts):
                projected_data = self.projections_[i, :].dot(X.T)

                inds = np.searchsorted(self.limits_[i][:self.n_bins_[i] - 1],
                                       projected_data, side='left')
                pred_scores[:, 0] += -self.weights[i] * np.log(
                    self.histograms_[i][inds])

        elif isinstance(self.n_bins, numbers.Integral):

            for i in range(self.num_random_cuts):
                projected_data = self.projections_[i, :].dot(X.T)
                # update the histograms
                self.histograms_[i, :], _ = np.histogram(
                    projected_data, bins=self.n_bins, density=False)
                self.histograms_[i, :] += 1e-12
                self.histograms_[i, :] /= np.sum(self.histograms_[i, :])

                inds = np.searchsorted(self.limits_[i, :self.n_bins - 1],
                                       projected_data, side='left')
                pred_scores[:, 0] += -self.weights[i] * np.log(
                    self.histograms_[i, inds])
        else:
            raise ValueError("n_bins must be an int or \'auto\', "
                             "got: %f" % self.n_bins)
        return self

    def score_partial(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Args:
            X numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns:
            anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        X = X.reshape(1, -1) if X.ndim == 1 else X
        X = check_array(X)
        pred_scores = np.zeros([X.shape[0], 1])
        if isinstance(self.n_bins, str) and self.n_bins.lower() == "auto":

            for i in range(self.num_random_cuts):
                projected_data = self.projections_[i, :].dot(X.T)
                inds = np.searchsorted(self.limits_[i][:self.n_bins_[i] - 1],
                                       projected_data, side='left')
                pred_scores[:, 0] += -self.weights[i] * np.log(
                    self.histograms_[i][inds])

        elif isinstance(self.n_bins, numbers.Integral):

            for i in range(self.num_random_cuts):
                projected_data = self.projections_[i, :].dot(X.T)

                inds = np.searchsorted(self.limits_[i, :self.n_bins - 1],
                                       projected_data, side='left')
                pred_scores[:, 0] += -self.weights[i] * np.log(
                    self.histograms_[i, inds])
        else:
            raise ValueError("n_bins must be an int or \'auto\', "
                             "got: %f" % self.n_bins)

        pred_scores /= self.num_random_cuts
        return pred_scores.ravel()