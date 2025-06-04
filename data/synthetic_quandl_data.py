import os
from typing import List

import numpy as np
import pandas as pd

from settings.default import QUANDL_TICKERS


def generate_synthetic_price_data(start_date: str, end_date: str, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    n = len(dates)

    mu = 0.0002  # daily drift
    sigma = 0.01  # daily volatility
    S0 = 100  # initial price

    returns = np.random.normal(loc=mu, scale=sigma, size=n)
    price = S0 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "Date": dates,
        "Settle": price
    })

    return df

def save_synthetic_quandl_data(tickers: List[str], output_dir: str, start_date: str, end_date: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, ticker in enumerate(tickers):
        df = generate_synthetic_price_data(start_date, end_date, seed=42 + i)
        file_name = os.path.join(output_dir, f"{ticker}.csv")

        df.to_csv(file_name, index=False)
        print(f"Saved synthetic data for {ticker} to {file_name}")


if __name__ == "__main__":
    OUTPUT_DIR = os.path.join("data", "quandl")
    START_DATE = "2016-01-01"
    END_DATE = "2022-01-01"

    save_synthetic_quandl_data(QUANDL_TICKERS, OUTPUT_DIR, START_DATE, END_DATE)
