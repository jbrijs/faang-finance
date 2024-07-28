import subprocess
import argparse

def run_script(script_path, ticker):
    subprocess.run(['python', script_path], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run stock data processing for a given ticker.')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    args = parser.parse_args()


    run_script('scripts/data_fetching.py', args.ticker)    
    run_script('scripts/data_normalization.py', args.ticker)
    run_script('scripts/feature_engineering.py', args.ticker)
    run_script('scripts/data_preprocessing.py', args.ticker)
    run_script('scripts/train_model.py', args.ticker)