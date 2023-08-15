import pytest

import pandas as pd
import numpy as np

from latent_calendar.const import TIME_SLOTS, DAYS_IN_WEEK
from latent_calendar.segments.hand_picked import (
    get_vocab_for_range,
    create_empty_template,
    create_hourly_segment,
    create_dow_segments,
    create_every_hour_segments,
)
from latent_calendar.vocab import DOWHour


def test_create_empty_template() -> None:
    df = create_empty_template()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (24, 7)


@pytest.mark.parametrize(
    "start, end, answer",
    [
        (DOWHour(dow=0, hour=0), DOWHour(dow=0, hour=1), ["00 00"]),
        (DOWHour(dow=0, hour=0), DOWHour(dow=0, hour=2), ["00 00", "00 01"]),
    ],
)
def test_get_vocab_for_range(start, end, answer) -> None:
    vocab = get_vocab_for_range(start=start, end=end)

    assert isinstance(vocab, list)
    assert vocab == answer


def df_strat_tests(df_strat: pd.DataFrame, index) -> None:
    assert isinstance(df_strat, pd.DataFrame)
    pd.testing.assert_index_equal(df_strat.index, pd.Index(index).sort_values())


@pytest.mark.parametrize(
    "start, end, answer",
    [
        # Single Day
        (DOWHour(dow=0, hour=0), DOWHour(dow=0, hour=1), 1),
        (DOWHour(dow=0, hour=0), DOWHour(dow=0, hour=24), 24),
        # Multi Day
        (DOWHour(dow=0, hour=23), DOWHour(dow=1, hour=1), 2),
        # Over Sunday
        (DOWHour(dow=6, hour=23), DOWHour(dow=0, hour=1), 2),
    ],
)
def test_create_hour_strat(start, end, answer) -> None:
    ser = create_hourly_segment(start=start, end=end, name="Test Name")

    assert isinstance(ser, pd.Series)
    assert ser.dtype == np.dtype("int")
    assert ser.sum() == answer
    assert ser.name == "Test Name"


def test_create_dow_strategy() -> None:
    df_strat = create_dow_segments()

    assert isinstance(df_strat, pd.DataFrame)
    assert df_strat.shape == (DAYS_IN_WEEK, TIME_SLOTS)


def test_create_every_hour_strategy() -> None:
    df_strat = create_every_hour_segments()

    assert isinstance(df_strat, pd.DataFrame)
    assert df_strat.shape == (TIME_SLOTS, TIME_SLOTS)
