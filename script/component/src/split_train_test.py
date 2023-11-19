# Import
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
    help="Output datastore folder URI",
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


# Function that reads the data
def get_data(data_path: str) -> tuple[np.ndarray, np.ndarray]:
    global X_files, y_files
    X_files = glob.glob(data_path + "/*-X.csv")
    y_files = glob.glob(data_path + "/*-y.csv")
    df_X = pd.concat(
        (pd.read_csv(f, header=0, index_col=0) for f in X_files), sort=False
    )
    df_y = pd.concat(
        (pd.read_csv(f, header=0, index_col=None) for f in y_files), sort=False
    )

    return df_X.values, df_y.values


# Function that split the dataset into X and y
def split_train_test(
    X_raw: pd.DataFrame,
    y_raw: pd.DataFrame,
    test_size: float = None,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=test_size, shuffle=shuffle
    )

    return X_train, X_test, y_train, y_test


# Main function
def main():
    # Read the data
    df_X, df_y = get_data(args.input_data)

    # Split the dataset into X and y
    X_train, X_test, y_train, y_test = split_train_test(
        df_X, df_y, test_size=args.test_size, shuffle=args.shuffle
    )

    # Save the data as a csv
    pd.DataFrame(X_train).to_csv(
        Path(args.output_data)
        / f"{Path(X_files[0]).name.replace('-X.csv','-X-train.csv')}",
        index=False,
        header=False,
    )
    pd.DataFrame(X_test).to_csv(
        Path(args.output_data)
        / f"{Path(X_files[0]).name.replace('-X.csv','-X-test.csv')}",
        index=False,
        header=False,
    )
    pd.DataFrame(y_train).to_csv(
        Path(args.output_data)
        / f"{Path(y_files[0]).name.replace('-y.csv','-y-train.csv')}",
        index=False,
        header=False,
    )
    pd.DataFrame(y_test).to_csv(
        Path(args.output_data)
        / f"{Path(y_files[0]).name.replace('-y.csv','-y-test.csv')}",
        index=False,
        header=False,
    )


if __name__ == "__main__":
    # Parse args
    args = parser.parse_args()

    # Run main function
    main()
