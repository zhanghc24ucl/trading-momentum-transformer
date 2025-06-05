import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mom_trans.backtest import run_classical_methods

INTERVALS = [(1990, y, y + 1) for y in range(2020, 2021)]

REFERENCE_EXPERIMENT = "experiment_yfinance_4assets_lstm_cpnone_len63_notime_div_v1"

features_file_path = os.path.join(
    "data",
    "yfinance_cpd_nonelbw.csv",
)

run_classical_methods(features_file_path, INTERVALS, REFERENCE_EXPERIMENT)
