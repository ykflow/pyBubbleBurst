import os
import numpy as np
import pandas as pd

def detrend_linear(y: np.ndarray) -> np.ndarray:
    """Removes a linear trend line from a 1D time-series.

    Args:
        y (np.ndarray): The raw 1D input time-series data.

    Returns:
        np.ndarray: The stationary, detrended residual vector.
    """
    y_arr = np.asarray(y, dtype=np.float64).flatten()
    T = len(y_arr)
    t = np.arange(T, dtype=np.float64)
    t_mean = t.mean()
    y_mean = y_arr.mean()

    t_cov = t - t_mean
    slope = np.dot(t_cov, y_arr - y_mean) / np.dot(t_cov, t_cov)
    intercept = y_mean - slope * t_mean
    trend = slope * t + intercept
    return y_arr - trend


def load_btc_data(file_path: str) -> pd.DataFrame:
    """
    Loads FRED formatted CBBTCUSD CSV files, handles date indices,
    and fills missing historical values (e.g., periods containing drops/periods like '.').
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target data payload missing at: {file_path}")

    df = pd.read_csv(file_path)
    df = df.rename(columns={'observation_date': 'Date', 'CBBTCUSD':'BTC/USD'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df['BTC/USD'] = pd.to_numeric(df['BTC/USD'], errors='coerce')
    df['BTC/USD'] = df['BTC/USD'].ffill().bfill()
    return df