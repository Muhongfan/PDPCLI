import argparse

def main():
    parser = argparse.ArgumentParser(prog='pdpcli', description= "Piggy's data processing CLI (PDPCLI) utility")

    parser.add_argument("-d", "--data", type = str, help="Path to the dataset (CSV format)")

    args = parser.parse_args()
    if args.data:
        print(f"Dataset provided: {args.data}")
    else:
        print("No dataset provided. Use --data to specify the path.")



main()