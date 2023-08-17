"""Example datasets for latent_calendar. 

"""

import pandas as pd

__all__ = ["load_online_transactions", "load_chicago_bikes"]


BASE_URL = "https://raw.githubusercontent.com/wd60622/latent-calendar/main/datasets"


def load_online_transactions() -> pd.DataFrame:
    """Kaggle Data for an non-store online retailer in UK. More information [here](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci).

    Returns:
        Online transactions data from a non-store online retailer in UK.

    """
    raise NotImplementedError("This dataset is not yet available.")
    file = f"{BASE_URL}/online-transaction-sbu.csv"

    return pd.read_csv(file, parse_dates=["InvoiceDate"], index_col=["Invoice"])


def load_chicago_bikes() -> pd.DataFrame:
    """Bikesharing trip level data from Chicago's Divvy system.

    Read more about the data source [here](https://data.cityofchicago.org/Transportation/Divvy-Trips/fg6s-gzvg).

    Returns:
        Trips data from Chicago's Divvy system.

    """
    file = f"{BASE_URL}/chicago-bikes.csv"

    return pd.read_csv(
        file, parse_dates=["started_at", "ended_at"], index_col=["ride_id"]
    )
