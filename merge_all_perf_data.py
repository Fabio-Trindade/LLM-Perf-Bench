import argparse
import os
import sys
import pandas as pd


def find_csv_files(base_dir: str, prefix: str):
    matched_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.startswith(prefix) and f.endswith(".csv"):
                matched_files.append(os.path.join(root, f))
    return sorted(matched_files)


def main():
    parser = argparse.ArgumentParser(
        description="Merge all CSV files matching a prefix into a single CSV"
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Base directory to search recursively"
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Filename prefix to match (e.g. perf_data.csv)"
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path to the merged output CSV"
    )

    args = parser.parse_args()

    csv_files = find_csv_files(args.dir, args.prefix)

    if not csv_files:
        print(f"ERROR: No CSV files found with prefix '{args.prefix}' in {args.dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files to merge")

    dataframes = []
    base_columns = None

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            if df.empty:
                print(f"WARNING: Skipping empty file {csv_file}")
                continue

            if base_columns is None:
                base_columns = list(df.columns)
            else:
                if list(df.columns) != base_columns:
                    raise ValueError(
                        f"Column mismatch in {csv_file}\n"
                        f"Expected: {base_columns}\n"
                        f"Found:    {list(df.columns)}"
                    )

            dataframes.append(df)
            print(f"Loaded: {csv_file} ({len(df)} rows)")

        except Exception as e:
            print(f"ERROR processing {csv_file}: {e}")
            sys.exit(2)

    if not dataframes:
        print("ERROR: No valid CSV files to merge")
        sys.exit(3)

    merged_df = pd.concat(dataframes, ignore_index=True)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    merged_df.to_csv(args.output_file, index=False)

    print(f"Merged CSV saved to: {args.output_file}")
    print(f"Total rows: {len(merged_df)}")


if __name__ == "__main__":
    main()
