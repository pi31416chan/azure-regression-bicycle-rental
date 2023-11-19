# Import
import argparse

import mlflow
import mltable
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Setup arg parser
parser = argparse.ArgumentParser()


# Add arguments
parser.add_argument(
    "--input_data",
    dest="input_data",
    type=str,
    help="Input dataset URI",
)
parser.add_argument(
    "--output_data",
    dest="output_data",
    type=str,
    help="Output MLflow model",
)
parser.add_argument(
    "--drop_col",
    dest="drop_col",
    type=str,
    help="Columns to be dropped, separated by comma",
)
parser.add_argument(
    "--label",
    dest="label",
    type=str,
    help="Column to split as the label, y",
)
parser.add_argument(
    "--test_size",
    dest="test_size",
    type=float,
    default=None,
    help=(
        "Float between 0.0 and 1.0 and represent the proportion of the "
        "dataset to include in the test split. If None, default to 0.25"
    ),
)
parser.add_argument(
    "--shuffle",
    dest="shuffle",
    type=bool,
    default=True,
    help=(
        "Whether or not to shuffle the data before splitting. Default to True"
    ),
)
parser.add_argument(
    "--alpha",
    dest="alpha",
    type=float,
    default=1.0,
    help=(
        "Constant that multiplies the L2 term, controlling regularization "
        "strength. alpha must be a non-negative float i.e. in [0, inf)"
    ),
)
parser.add_argument(
    "--fit_intercept",
    dest="fit_intercept",
    type=bool,
    default=True,
    help="Whether to calculate the intercept for this model",
)
parser.add_argument(
    "--tol",
    dest="tol",
    type=float,
    default=1e-4,
    help=("Tolerance for stopping criteria. Default to 1e-4"),
)


# Setup MLflow autolog
mlflow.autolog(log_datasets=False)


# Function that reads the data
def get_data() -> pd.DataFrame:
    df = mltable.load(args.input_data).to_pandas_dataframe()

    return df


# Function that drop the columns
def drop_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    column_to_drop = args.drop_col.split(",")
    df = dataframe.drop(columns=column_to_drop)

    return df


# Function that split the dataset into X and y
def split_x_y(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_X = dataframe.drop(columns=args.label)
    df_y = dataframe[[args.label]]

    return df_X, df_y


# Function that split the dataset into X and y
def split_train_test(
    X_raw: pd.DataFrame, y_raw: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=args.test_size, shuffle=args.shuffle
    )

    return X_train, X_test, y_train, y_test


# Function that trains the model
def train_and_evaluate_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    # Ridge Regression model
    model = Ridge(
        alpha=args.alpha, fit_intercept=args.fit_intercept, tol=args.tol
    )
    model.fit(X_train, y_train)

    # Scoring the model
    test_score_r2 = model.score(X_test, y_test)

    # Dump and log the model
    # Comment this line for making into Azure ML component, uncomment for local
    # script testing
    # from pathlib import Path
    # from shutil import rmtree
    # model_path = Path(args.output_data)
    # if model_path.exists():
    #     rmtree(model_path)
    mlflow.sklearn.save_model(model, args.output_data)


# Main function
def main():
    # Read the data
    df = get_data()

    # Drop unwanted columns
    df = drop_column(df)

    # Split dataset into X and y
    df_X, df_y = split_x_y(df)

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = split_train_test(df_X, df_y)

    # Train, evaluate and save the model
    train_and_evaluate_model(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    # Parse args
    args = parser.parse_args()

    # Run main function
    main()
