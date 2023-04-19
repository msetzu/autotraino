from __future__ import annotations

import os.path
from typing import Optional, Sequence

import numpy
from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.tabular import TabularDataset, TabularPredictor
from pandas import DataFrame

from .. import Trainer


class AutogluonTrainer(Trainer):
    def __init__(self, save_path: Optional[str] = "./"):
        """An AutoGluon autotrainer.
        Args:
            save_path: Path where to store the trainer's output. Defaults to the current directory.
                       Note that, in order to avoid default overwrite
                       (see https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html),
                       we create one folder per run of this Trainer inside `save_path`. Effectively,
                       the output will be stored in '{save_path}/{i}', with {i} incremented by one at each new
                       call to fit.
                       Folder is created if it does not exist.
        """
        super(AutogluonTrainer, self).__init__()
        self.models_hyperparameters = dict()
        self.trainer = None
        self.names = list()
        self.target_feature = None

        self.save_path = save_path
        self.run_paths = list()

        if not os.path.exists(save_path):
            os.mkdir(save_path)

    def fit(self, dataset: DataFrame, validation_dataset: Optional[DataFrame] = None,
            task: str = "binary", target_feature: str = "", **kwargs) -> Trainer:
        """Fit this trainer to the given `dataset`.

        Args:
            dataset: The dataset to train on.
            validation_dataset: Optional validation dataset.
            task: "binary", "multiclass", or "regression". Defaults to "binary". Adjust according to your task.
            target_feature: Name of the target feature to predict.
            kwargs: Keyword arguments fed to the trainer:
                save_folder (str): Folder where to store the models.
                metric (str): Evaluation metric, check
                              https://auto.gluon.ai/stable/api/autogluon.tabular.models.html#abstractmodel
                              for details. Defaults to "accuracy".
                time_limit (int): Time (seconds) budget of the trainer.
                sample_weight (Series, default = None): The training data sample weights.
                num_cpus (int, default = 'auto'): How many CPUs to use during fit. This is counted in virtual cores,
                                                  not in physical cores. If ‘auto’, model decides.
                num_gpus (int, default = 'auto'): How many GPUs to use during fit. If ‘auto’, model decides.

        Returns:
            This trainer, fit on `dataset`.
        """
        autogluon_feature_map = {
            f: dataset.dtypes[f].name.replace("8", "").replace("16", "").replace("32", "").replace("64", "")
            for f in dataset.columns
        }
        autogluon_feature_map = FeatureMetadata(type_map_raw=autogluon_feature_map)
        gluon_data = TabularDataset(dataset)
        self.trainer = TabularPredictor(problem_type=task, label=target_feature,
                                        path=f"{self.save_path}/{len(self.run_paths)}")
        self.trainer = self.trainer.fit(train_data=gluon_data,
                                        tuning_data=validation_dataset,
                                        feature_metadata=autogluon_feature_map,
                                        **kwargs)

        # extract predictors
        models_infos = self.trainer.info()["model_info"]
        self.names = list(models_infos.keys())

        # models._learner
        for name in self.names:
            actual_model = self.trainer._trainer.load_model(model_name=name)
            self.trained_models[name] = actual_model
            self.models_hyperparameters[name] = models_infos[name]["hyperparameters"]

        self.run_paths.append(len(self.run_paths))
        self.target_feature = target_feature
        self.is_fit = True

        return self

    def predict(self, data: DataFrame, with_models: Optional[str | Sequence[str]] = None) -> numpy.ndarray:
        """Predict on the given data, optionally on a subset of models.

        Args:
            data: The data to predict on.
            with_models: Models to make the prediction with. Defaults to None (use all models).

        Returns:
            Predictions.
        """
        if not self.is_fit:
            raise ValueError("Train the Trainer first.")

        if with_models is None:
            models_to_use = self.names
        elif isinstance(with_models, str):
            models_to_use = [with_models]
        else:
            models_to_use = list(with_models)

        if self.target_feature in data.columns:
            processed_data = data.drop(self.target_feature, axis="columns", inplace=False)
        else:
            processed_data = data
        predictions = self.trainer.predict_multi(processed_data, models=models_to_use)
        predictions = numpy.vstack([predictions[m].values for m in models_to_use])

        return predictions

    def predict_proba(self, data: DataFrame, with_models: Optional[str | Sequence[str]] = None) -> numpy.ndarray:
        """Predict on the given data, optionally on a subset of models.

        Args:
            data: The data to predict on.
            with_models: Models to make the prediction with. Defaults to None (use all models).

        Returns:
            Predictions.
        """
        if not self.is_fit:
            raise ValueError("Train the Trainer first.")

        if with_models is None:
            models_to_use = self.names
        elif isinstance(with_models, str):
            models_to_use = [with_models]
        else:
            models_to_use = list(with_models)

        if self.target_feature in data.columns:
            processed_data = data.drop(self.target_feature, axis="columns", inplace=False)
        else:
            processed_data = data
        predictions = self.trainer.predict_proba_multi(processed_data, models=models_to_use)
        predictions = numpy.array([predictions[m].values for m in models_to_use])

        return predictions

    @staticmethod
    def load(path: str) -> AutogluonTrainer:
        """Load an AutogluonTrainer from the given path

        Args:
            path: Path to the Autogluon storage folder.

        Returns:
            An AutogluonTrainer.
        """
        trainer = AutogluonTrainer()
        trainer.trainer = TabularPredictor.load(path)
        trainer.is_fit = True

        models_infos = trainer.trainer.info()["model_info"]
        trainer.names = list(models_infos.keys())

        for name in trainer.names:
            actual_model = trainer.trainer._trainer.load_model(model_name=name)
            trainer.trained_models[name] = actual_model
            trainer.models_hyperparameters[name] = models_infos[name]["hyperparameters"]

        return trainer

    def __getitem__(self, item):
        return self.trainer._trainer.load_model(item)

    def __iter__(self):
        for name in self.names:
            yield self[name].model
