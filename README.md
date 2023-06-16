# Autotraino :truck:

`autotraino` is a small wrapper library for AutoML on tabular datasets.

```python
from autotraino.gluon import AutogluonTrainer 
from datasets import load_dataset

train = load_dataset("mstz/adult", "income")["train"].to_pandas()

# train the model
trainer = AutogluonTrainer()
trainer = trainer.fit(train, target_feature="over_threshold", time_limit=100)
```
When fitting we can control basic parameters such as where to store the resulting models
(parameter `save_path` of the trainer constructor) or the time budget assigned to the trainer (parameter
`time_limit`, expressed in seconds).

Once trained, we can access the single models
```python
# trained models
print(trainer.names)

print(trainer["LightGBM"])
```
and predict directly from the `Trainer` itself:
```python
train_x = train.copy().drop("over_threshold", axis="columns")
predictions = trainer.predict(train_x, with_models=["LightGBM", "RandomForest"])
```
`predict` yields an `m x n` numpy array, where each of the `m` rows holds the predictions of a different model.
Models can be chosen with the parameter `with_models`, if not provided all models are used by default.


# Quickstart
You can install `autotraino` via pypi (strongly recommended to create a virtual environment, see [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)):
```shell
pip3.10 install autotraino
```
By default, no trainers are installed, to install one use one of
```shell
# basic, does not include neural models, catboost, and lightgbm
pip install autotraino[autogluon]
```
or
```shell
# includes everything
pip install autotraino[autogluon_all]
```

# Datasets
`autotraino` is based off `pandas.DataFrame`s.
You can find a large collection on a Huggingface repository I'm curating at [huggingface.co/mstz](https://huggingface.co/mstz).
Datasets are sourced from UCI, Kaggle, and OpenML.
Several are still to be updated (especially dataset cards).

## What model families to train?
Currently based on [Autogluon](https://auto.gluon.ai/stable/index.html), `autotraino` currently trains the following models:
- Boosting models
  - LightGBM
  - CatBoost
- Bagging models
  - Random Forest
  - ExtraTree Classifier
- Neural Network
  - FastAI
  - NNTorch
- Classical AI models
  - k-NN
  - Logistic Regression

## Preprocessing
`autotraino` automatically detects feature types and performs the necessary feature preprocessing per model.
To ease the process, consider setting the appropriate `dtypes` in the input `pandas.DataFrame`.

# In the works
Future developments include:
- Fitting arbitrary functions (ray tune)
- Fitting multi-output models.