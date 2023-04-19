"""Autotraining models."""
from __future__ import annotations

from abc import ABC, abstractmethod
import sys
from typing import Optional, Sequence

import numpy
from pandas import DataFrame

sys.path.append("../../datati/src/")


class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, data, **kwargs):
        """Predict on the given `data`. Optional parameters to be fed as keyword arguments.

        Args:
            data: The input data.
            **kwargs: Optional keyword arguments.

        Returns:
            The model's predictions.
        """
        pass


class Trainer(ABC):
    """Train a (set of) model(s) on the given data.

    Attributes:
        is_fit: True if this trainer has been fit, False otherwise.
        trained_models: Trained models.
    """
    def __init__(self):
        self.is_fit = False
        self.trained_models = dict()

    @abstractmethod
    def fit(self, dataset: DataFrame, storage_folder: str, **trainer_kwargs) -> Trainer:
        """Fit this trainer to the given `dataset`.

        Args:
            dataset: The dataset to train on.
            storage_folder: Folder where to store the model files.
            trainer_kwargs: Keyword arguments fed to the trainer.

        Returns:
            This trainer, fit on `dataset`.
        """
        pass

    @abstractmethod
    def predict(self, data: DataFrame, with_models: Optional[str | Sequence[str]] = None) -> numpy.ndarray:
        """Predict on the given data, optionally on a subset of models.

        Args:
            data: The data to predict on.
            with_models: Models to make the prediction with. Defaults to None (use all models).

        Returns:
            Predictions.
        """
        pass
