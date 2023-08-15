"""Example datasets for latent_calendar. 

"""
from pathlib import Path

import pandas as pd

__all__ = ["load_online_transactions", "load_chicago_bikes"]


HERE = Path(__file__).parent


def load_online_transactions() -> pd.DataFrame:
    """Kaggle Data for an non-store online retailer in UK. More information [here](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)."""
    file = HERE / "online_retail_II.csv"

    return pd.read_csv(file, parse_dates=["InvoiceDate"], index_col=["Invoice"])


def load_chicago_bikes() -> pd.DataFrame:
    """Bikesharing trip level data from Chicago's Divvy system.

    Read more about the data source [here](https://data.cityofchicago.org/Transportation/Divvy-Trips/fg6s-gzvg).

    Pulled

    Returns:
        Trips data from Chicago's Divvy system.

    """
    file = HERE / "chicago-bikes.csv"

    return pd.read_csv(
        file, parse_dates=["started_at", "ended_at"], index_col=["ride_id"]
    )
