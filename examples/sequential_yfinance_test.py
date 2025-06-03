import argparse
import os
import datetime as dt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from settings.default import (
    YFINANCE_TICKERS,
    CPD_YFINANCE_OUTPUT_FOLDER,
    CPD_DEFAULT_LBW,
    USE_KM_HYP_TO_INITIALISE_KC
)
import mom_trans.changepoint_detection as cpd
from mom_trans.data_prep import calc_returns
from data.pull_data import pull_yfinance_data

def main(lookback_window_length: int, test_mode: bool = False):
    output_folder = CPD_YFINANCE_OUTPUT_FOLDER(lookback_window_length)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    start_date = dt.datetime.strptime("1990-01-01", "%Y-%m-%d")
    end_date = dt.datetime.strptime("2021-12-31", "%Y-%m-%d")
    
    # In test mode, only process first 3 tickers
    tickers_to_process = YFINANCE_TICKERS[:3] if test_mode else YFINANCE_TICKERS
    
    print(f"Processing {len(tickers_to_process)} tickers sequentially...")
    if test_mode:
        print("TEST MODE: Only processing first 3 tickers")
    
    results = []
    for i, ticker in enumerate(tickers_to_process, 1):
        try:
            print(f"[{i}/{len(tickers_to_process)}] Processing {ticker}...")
            
            # Load data
            data = pull_yfinance_data(ticker)
            data["daily_returns"] = calc_returns(data["close"])
            
            # Create output path
            filename = ticker.replace("=", "_").replace("^", "_").replace(".", "_") + ".csv"
            output_file_path = os.path.join(output_folder, filename)
            
            # Run changepoint detection
            cpd.run_module(
                data, 
                lookback_window_length, 
                output_file_path, 
                start_date, 
                end_date, 
                USE_KM_HYP_TO_INITIALISE_KC
            )
            
            print(f"✓ Completed {ticker}")
            results.append(f"Success: {ticker}")
            
        except Exception as e:
            error_msg = f"Error processing {ticker}: {str(e)}"
            print(f"✗ {error_msg}")
            results.append(error_msg)
    
    # Print summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY:")
    print("="*50)
    
    successes = [r for r in results if r.startswith("Success")]
    errors = [r for r in results if r.startswith("Error")]
    
    print(f"Successful: {len(successes)}")
    print(f"Failed: {len(errors)}")
    
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")

if __name__ == "__main__":
    def get_args():
        parser = argparse.ArgumentParser(
            description="Run changepoint detection sequentially for yfinance tickers"
        )
        parser.add_argument(
            "lookback_window_length",
            metavar="l",
            type=int,
            nargs="?",
            default=CPD_DEFAULT_LBW,
            help="CPD lookback window length",
        )
        parser.add_argument(
            "--test",
            action="store_true",
            help="Test mode: only process first 3 tickers"
        )
        
        args = parser.parse_known_args()[0]
        return args.lookback_window_length, args.test

    main(*get_args())