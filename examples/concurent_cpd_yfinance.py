import multiprocessing
import argparse
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from settings.default import (
    QUANDL_TICKERS,
    CPD_QUANDL_OUTPUT_FOLDER,
    CPD_DEFAULT_LBW,
)

#--------------------#
# Temporary Solution #
#--------------------#
YFINANCE_TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

CPD_YFINANCE_OUTPUT_FOLDER = lambda lbw: os.path.join(
    "data", f"yfinance_cpd_{(lbw if lbw else 'none')}lbw"
)
CPD_YFINANCE_OUTPUT_FOLDER_DEFAULT = CPD_YFINANCE_OUTPUT_FOLDER(CPD_DEFAULT_LBW)

FEATURES_YFINANCE_FILE_PATH = lambda lbw: os.path.join(
    "data", f"yfinance_cpd_{(lbw if lbw else 'none')}lbw.csv"
)
FEATURES_YFINANCE_FILE_PATH_DEFAULT = FEATURES_YFINANCE_FILE_PATH(CPD_DEFAULT_LBW)
#--------------------#


N_WORKERS = len(YFINANCE_TICKERS)


def main(lookback_window_length: int):
    if not os.path.exists(CPD_YFINANCE_OUTPUT_FOLDER(lookback_window_length)):
        os.mkdir(CPD_YFINANCE_OUTPUT_FOLDER(lookback_window_length))

    all_processes = [
        f'python -m examples.cpd_yfinance "{ticker}" "{os.path.join(CPD_YFINANCE_OUTPUT_FOLDER(lookback_window_length), ticker + ".csv")}" "2019-01-01" "2021-12-31" "{lookback_window_length}"'
        for ticker in YFINANCE_TICKERS
    ]
    process_pool = multiprocessing.Pool(processes=N_WORKERS)
    process_pool.map(os.system, all_processes)


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        parser = argparse.ArgumentParser(
            description="Run changepoint detection module for all tickers"
        )
        parser.add_argument(
            "lookback_window_length",
            metavar="l",
            type=int,
            nargs="?",
            default=CPD_DEFAULT_LBW,
            help="CPD lookback window length",
        )
        return [
            parser.parse_known_args()[0].lookback_window_length,
        ]

    main(*get_args())
