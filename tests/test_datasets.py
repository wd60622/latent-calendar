import os
from pathlib import Path

import pytest

import pandas as pd

from latent_calendar.datasets import (
    load_chicago_bikes,
    load_online_transactions,
    load_ufo_sightings,
    HERE,
)


@pytest.mark.skipif(
    os.environ.get("CI") != "INTEGRATION", reason="CI is not set to INTEGRATION"
)
@pytest.mark.parametrize(
    "load_func, local_file",
    [
        (load_chicago_bikes, "chicago-bikes.csv"),
        (load_online_transactions, "online_retail_II.csv"),
        (load_ufo_sightings, "ufo_sighting_data.csv"),
    ],
)
def test_load_func(load_func, local_file):
    file: Path = HERE / local_file
    file.unlink(missing_ok=True)

    df = load_func(local_save=True)
    df_second_time = load_func()

    pd.testing.assert_frame_equal(df, df_second_time)
