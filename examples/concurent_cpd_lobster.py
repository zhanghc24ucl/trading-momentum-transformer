import multiprocessing
import argparse
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from settings.default import (
    CPD_DEFAULT_LBW,
)

#--------------------#
# Temporary Solution #
#--------------------#

LOBSTER_TICKERS = ['AAPL']

CPD_LOBSTER_OUTPUT_FOLDER = lambda lbw: os.path.join(
    "data", f"lobster_cpd_{(lbw if lbw else 'none')}lbw"
)

FEATURES_LOBSTER_FILE_PATH = lambda lbw: os.path.join(
    "data", f"lobster_cpd_{(lbw if lbw else 'none')}lbw.csv"
)

#--------------------#

N_WORKERS = min(len(LOBSTER_TICKERS), multiprocessing.cpu_count())


def main(lookback_window_length: int):
    if not os.path.exists(CPD_LOBSTER_OUTPUT_FOLDER(lookback_window_length)):
        os.makedirs(CPD_LOBSTER_OUTPUT_FOLDER(lookback_window_length))

    all_processes = [
        f'python -m examples.cpd_lobster "{ticker}" "{os.path.join(CPD_LOBSTER_OUTPUT_FOLDER(lookback_window_length), ticker.replace("=", "_").replace("^", "_").replace(".", "_") + ".csv")}" "1990-01-01" "2025-12-31" "{lookback_window_length}"'
        for ticker in LOBSTER_TICKERS
    ]
    process_pool = multiprocessing.Pool(processes=N_WORKERS)
    process_pool.map(os.system, all_processes)
    process_pool.close()
    process_pool.join()


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        parser = argparse.ArgumentParser(
            description="Run changepoint detection module for all yfinance tickers"
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
    