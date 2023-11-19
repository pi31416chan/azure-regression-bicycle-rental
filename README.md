# Azure ML - Regresssion for Bicycle Rentals
Repository for the project of facilitating Azure ML for training a regression 
machine learning model to predict bicycle rentals for a given day.

## Dataset
name: [daily-bike-share.csv](data/daily-bike-share.csv) \
Type: `MLTable`

## Model
1. Linear Regression (as Azure ML pipeline component)
2. Ridge Regression (as Azure ML command job for grid search/hyperparameter 
tuning)

## Azure ML Component
### Drop Columns
Drop specified columns from input dataset.

Inputs:
|Parameter|Type|Mode|Description|
|-|-|-|-|
|input_data|mltable|direct|Input dataset URI|
|drop_col|string|NA|Columns to be dropped, separated by comma|

Outputs:
|Parameter|Type|Mode|Description|
|-|-|-|-|
|dropped_col_data|uri_folder|rw_mount|Output datastore folder URI|

### Split X y
Split the dataset into X and y with the given y label.

Inputs:
|Parameter|Type|Mode|Description|
|-|-|-|-|
|input_data|uri_folder|ro_mount|Input datastore folder URI|
|label|string|NA|Column to split as the label, y|

Outputs:
|Parameter|Type|Mode|Description|
|-|-|-|-|
|x_y_data|uri_folder|rw_mount|Output datastore folder URI|

### Split Train Test
Split the Xy dataset into X_train, X_test, y_train, y_test according to
specified proportion.

Inputs:
|Parameter|Type|Mode|Description|
|-|-|-|-|
|input_data|uri_folder|ro_mount|Input datastore folder URI|
|test_size|number|NA|Float between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If None, default to 0.25|
|shuffle|boolean|NA|Whether or not to shuffle the data before splitting. Default to True|

Outputs:
|Parameter|Type|Mode|Description|
|-|-|-|-|
|train_test_data|uri_folder|rw_mount|Output datastore folder URI|

### Linear Regression
Train a linear regression model and evaluate its performance.

Inputs:
|Parameter|Type|Mode|Description|
|-|-|-|-|
|input_data|uri_folder|ro_mount|Input datastore folder URI|
|fit_intercept|boolean|NA|Whether to calculate the intercept for this model. Default to True|

Outputs:
|Parameter|Type|Mode|Description|
|-|-|-|-|
|l_reg_model|mlflow_model|rw_mount|Output MLflow model|

## Azure ML Pipeline
A pipeline to train the model with several intermediate steps from data 
cleansing to training the model, pipeline behavior can be altered with 
various parameters.

Pipeline Composition: \
<img width="143" alt="pipeline" src="https://github.com/pi31416chan/azure-regression-bicycle-rental/assets/89824274/59aaf3c2-39b3-4e7d-a0ce-9b8f4f0ba418">

## Azure ML Job
### Train Ridge Regression
Job to train a Ridge Regression model to predict bicycle rentals.

Inputs:
|Parameter|Type|Mode|
|-|-|-|
|input_data|mltable|direct|
|drop_col|string|NA|
|label|string|NA|
|test_size|number|NA|
|shuffle|boolean|NA|
|alpha|number|NA|
|fit_intercept|boolean|NA|
|tol|number|NA|

Outputs:
|Parameter|Type|Mode|
|-|-|-|
|ridge_reg_model|mlflow_model|rw_mount|

## Azure ML Sweep Job
### Ridge Regression Sweep
Example:
```
ridge_reg_job_for_sweep = ridge_reg_job(
    test_size=Choice(values=[0.20, 0.25]),
    alpha=Choice(values=[1.0, 5.0, 10.0]),
)
ridge_reg__sweep = ridge_reg_job_for_sweep.sweep(
    sampling_algorithm="grid",
    primary_metric="Ridge_score_X_test",
    goal="Maximize",
)
ridge_reg__sweep.set_limits(max_concurrent_trials=2)
```

Example Result: \
<img width="285" alt="image" src="https://github.com/pi31416chan/azure-regression-bicycle-rental/assets/89824274/06694c7f-3336-4647-b23e-48e62040441d">

## Azure ML Environment
### basic
Environment for data preparation, designed as lightweight as possible
for quick spinning up of cloud compute resource.

Name: basic \
Container: `az-basic:latest` \
YAML: [az-basic-conda.yaml](script/environment/az-basic-conda.yaml)

### sklearn
Environment for machine learning with scikit-learn, container is built
with the previous **basic** container image as base image.

Name: basic \
Container: `az-sklearn:latest` \
YAML: [az-sklearn-conda.yaml](script/environment/az-sklearn-conda.yaml)

## Conclusion
Ridge Regression with certain degree of regularization performs better in 
term of R2 score compared to Linear Regression.

Same steps can be replicated for other algorithms to search for the best
algorithm to predict this dataset.
