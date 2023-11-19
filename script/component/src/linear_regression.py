# Import
import argparse
import glob

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Setup arg parser
parser = argparse.ArgumentParser()


# Add arguments
parser.add_argument(
    "--input_data",
    dest="input_data",
    type=str,
    help="Input datastore folder URI",
)
parser.add_argument(
    "--output_data",
    dest="output_data",
    type=str,
    help="Output MLflow model",
)
parser.add_argument(
    "--fit_intercept",
    dest="fit_intercept",
    type=bool,
    default=True,
    help="Whether to calculate the intercept for this model",
)


# Setup MLflow autolog
mlflow.autolog()


# Function that reads the data
def get_data(
    data_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global X_files, y_files
    X_train_files = glob.glob(data_path + "/*-X-train.csv")
    X_test_files = glob.glob(data_path + "/*-X-test.csv")
    y_train_files = glob.glob(data_path + "/*-y-train.csv")
    y_test_files = glob.glob(data_path + "/*-y-test.csv")
    X_train = pd.read_csv(X_train_files[0], header=None, index_col=None)
    X_test = pd.read_csv(X_test_files[0], header=None, index_col=None)
    y_train = pd.read_csv(y_train_files[0], header=None)
    y_test = pd.read_csv(y_test_files[0], header=None)

    return X_train, X_test, y_train, y_test


# Main function
def main():
    # Read the data
    X_train, X_test, y_train, y_test = get_data(args.input_data)

    # Linear Regression model
    model = LinearRegression(fit_intercept=args.fit_intercept)
    model.fit(X_train, y_train)

    # Scoring the model
    # train_score_r2 = model.score(X_train, y_train)
    test_score_r2 = model.score(X_test, y_test)

    # Log model score
    # mlflow.log_metrics(
    #     {
    #         "train_score_r2": train_score_r2,
    #         "test_score_r2": test_score_r2,
    #     }
    # )

    # Dump and log the model
    # Comment this line for making into Azure ML component, uncomment for local
    # script testing
    # from pathlib import Path
    # from shutil import rmtree
    # model_path = Path(args.output_data)
    # if model_path.exists():
    #     rmtree(model_path)
    mlflow.sklearn.save_model(model, args.output_data)


if __name__ == "__main__":
    # Parse args
    args = parser.parse_args()

    # Log fit-intercept parameter
    # mlflow.log_param("fit_intercept", args.fit_intercept)

    # Run main function
    main()
