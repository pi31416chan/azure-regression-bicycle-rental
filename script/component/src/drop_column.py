# Import
import argparse
from pathlib import Path

import mltable

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
    help="Output datastore folder URI",
)
parser.add_argument(
    "--drop_col",
    dest="drop_col",
    type=str,
    help="Columns to be dropped, separated by comma",
)

# Parse args
args = parser.parse_args()

# Read the data
df = mltable.load(args.input_data).to_pandas_dataframe()

# Drop the columns
column_to_drop = args.drop_col.split(",")
df = df.drop(columns=column_to_drop)

# Save the data as a csv
df.to_csv(Path(args.output_data) / "dropped.csv", index=False)
