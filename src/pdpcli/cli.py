import argparse
from pdpcli.processor import recommend_preprocessing, apply_preprocessing, validate_processing
from pdpcli.evaluate import evaluate_processing

def main():
    parser = argparse.ArgumentParser(prog='pdpcli', description= "Piggy's data processing CLI (PDPCLI) utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # pdpcli recommend -d input.csv  -> R1 R2 R3...
    recommend_parser = subparsers.add_parser("recommend", help = "Recommend data preprocessing steps")
    recommend_parser.add_argument("-d", "--raw_data", required=True, help="Path to CSV dataset file (CSV format).")
    recommend_parser.add_argument("-t", "--target", default=None, help="Target column name (optional).")
    recommend_parser.add_argument("--time", default="timestamp", help="Timestamp column name (default: timestamp)")

    # pdpcli apply --data data.csv --steps R1 R3 --output processed.csv -> output.file
    apply_parser = subparsers.add_parser("apply", help="Apply preprocessing steps to dataset.")
    apply_parser.add_argument("-d", "--raw_data", required=True, help="Path to CSV dataset file (CSV format).")
    apply_parser.add_argument("-t", "--target", default=None, help="Target column name (optional).")
    apply_parser.add_argument("--steps", nargs="+", help="Preprocessing steps to apply (e.g. R1 R2 R4)")
    apply_parser.add_argument("--time", default="timestamp", help="Timestamp column name (default: timestamp)")
    apply_parser.add_argument("-o", "--output", type = str, required=True, help = "Path to output file (optional)")
    apply_parser.add_argument(
        "--auto-order",
        action="store_true",
        help="Apply data preprocessing steps in recommended execution order"
    )
    # pdpcli validate --data data.csv
    validate_parser = subparsers.add_parser("validate", help="Validate the correlations of the varialbes")
    validate_parser.add_argument("-d", "--raw_data", required=True, help="Path to CSV dataset file (CSV format).")
    validate_parser.add_argument("--time", default="timestamp", help="Timestamp column name (default: timestamp)")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate forecasting models on raw and processed data.")
    evaluate_parser.add_argument(
        "-d", "--raw_data",
        type=str,
        required=True,
        help="Path to the raw CSV file (e.g., without preprocessing)."
    )
    evaluate_parser.add_argument(
        "--processed_file",
        type=str,
        required=True,
        help="Path to the processed CSV file (with preprocessing applied)."
    )
    evaluate_parser.add_argument(
        "-t", "--target",
        type=str,
        required=True,
        help="Name of the target column to forecast."
    )
    evaluate_parser.add_argument(
        "--time",
        type=str,
        default="timestamp",
        help="Name of the timestamp column (default: 'timestamp')."
    )
    args = parser.parse_args()

    if args.command == "recommend":
        recommend_preprocessing(args.raw_data, args.time, args.target)

    elif args.command == "apply":
        apply_preprocessing(
        data_path=args.raw_data,
        steps=args.steps,
        output_path=args.output,
        time_col=args.time,
        target_col=args.target,
        auto_order=args.auto_order
    )
        
    elif args.command == "validate":
        validate_processing(
            data_path = args.raw_data,
            time_col= args.time
        )
    elif args.command == "evaluate":
        evaluate_processing(
        data_path=args.raw_data,
        processed_file=args.processed_file,
        target_col=args.target,
        time_col=args.time
    )

            

