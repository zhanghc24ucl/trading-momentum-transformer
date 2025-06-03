
import multiprocessing
import argparse
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from settings.default import (
    YFINANCE_TICKERS,
    CPD_YFINANCE_OUTPUT_FOLDER,
    CPD_DEFAULT_LBW,
)

N_WORKERS = min(len(YFINANCE_TICKERS), multiprocessing.cpu_count())


def main(lookback_window_length: int):
    if not os.path.exists(CPD_YFINANCE_OUTPUT_FOLDER(lookback_window_length)):
        os.makedirs(CPD_YFINANCE_OUTPUT_FOLDER(lookback_window_length))

    all_processes = [
        f'python -m examples.cpd_yfinance "{ticker}" "{os.path.join(CPD_YFINANCE_OUTPUT_FOLDER(lookback_window_length), ticker.replace("=", "_").replace("^", "_").replace(".", "_") + ".csv")}" "1990-01-01" "2021-12-31" "{lookback_window_length}"'
        for ticker in YFINANCE_TICKERS
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
    