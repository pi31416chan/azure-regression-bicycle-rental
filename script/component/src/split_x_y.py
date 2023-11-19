# Import
import argparse
import glob
from pathlib import Path

import pandas as pd

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
    "--label",
    dest="label",
    type=str,
    help="Column to split as the label, y",
)


# Function that reads the data
def get_data(data_path: str) -> pd.DataFrame:
    global all_files
    all_files = glob.glob(data_path + "/*.csv")
    df = pd.concat((pd.read_csv(f, header=0) for f in all_files), sort=False)

    return df


# Function that split the dataset into X and y
def split_x_y(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_X = dataframe.drop(columns=args.label)
    df_y = dataframe[[args.label]]

    return df_X, df_y


# Main function
def main():
    # Read the data
    df = get_data(args.input_data)

    # Split the dataset into X and y
    df_X, df_y = split_x_y(df)

    # Save the data as a csv
    df_X.to_csv(
        Path(args.output_data)
        / f"{Path(all_files[0]).name.replace('.csv','-X.csv')}",
        index=False,
    )
    df_y.to_csv(
        Path(args.output_data)
        / f"{Path(all_files[0]).name.replace('.csv','-y.csv')}",
        index=False,
    )


if __name__ == "__main__":
    # Parse args
    args = parser.parse_args()

    # Run main function
    main()
