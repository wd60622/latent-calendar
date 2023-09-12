import pytest

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import latent_calendar  # noqa
from latent_calendar.const import TIME_SLOTS, FULL_VOCAB, DAYS_IN_WEEK, HOURS_IN_DAY
from latent_calendar.segments.convolution import day_of_week_column_name_func


@pytest.fixture
def ser() -> pd.Series:
    return pd.Series(pd.date_range("2023-01-01", "2023-01-14", freq="H"))


def test_series_extensions(ser) -> None:
    assert hasattr(ser, "cal")

    ax = ser.cal.plot()
    assert isinstance(ax, plt.Axes)


@pytest.fixture
def df() -> pd.DataFrame:
    """Generate some fake data."""
    return pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    ).T


@pytest.fixture
def df_segments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment 1": [1, 0, 1],
            "segment 2": [0, 1, 0],
            "segment 3": [1, 1, 0],
        }
    ).T


def test_all_dataframe_extensions(df, df_segments) -> None:
    assert hasattr(df, "cal")

    pd.testing.assert_frame_equal(
        df.cal.normalize("max"),
        pd.DataFrame(
            {
                "a": [1 / 3, 2 / 3, 1],
                "b": [4 / 6, 5 / 6, 1],
            }
        ).T,
    )

    pd.testing.assert_frame_equal(
        df.cal.normalize("probs"),
        pd.DataFrame(
            {
                "a": [1 / 6, 2 / 6, 3 / 6],
                "b": [4 / 15, 5 / 15, 6 / 15],
            }
        ).T,
    )

    pd.testing.assert_frame_equal(
        df.cal.normalize("even_rate"),
        pd.DataFrame(
            {
                "a": [1 / 3, 2 / 3, 3 / 3],
                "b": [4 / 3, 5 / 3, 6 / 3],
            }
        ).T,
    )

    with pytest.raises(ValueError):
        df.cal.normalize("unknown")

    pd.testing.assert_frame_equal(
        df.cal.sum_over_segments(df_segments),
        pd.DataFrame(
            {
                "segment 1": [4, 10],
                "segment 2": [2, 5],
                "segment 3": [3, 9],
            },
            index=df.index,
        ),
    )


@pytest.fixture
def df_long(ser) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "group": rng.choice(["a", "b", "c"], size=len(ser)),
            "another_group": rng.choice(["x"], size=len(ser)),
            "timestamp": ser,
        }
    )


def test_long_dataframe_extensions(df_long) -> None:
    assert hasattr(df_long, "cal")

    df_wide = df_long.cal.aggregate_events("group", "timestamp")
    assert df_wide.shape == (3, TIME_SLOTS)

    df_wide = df_long.cal.aggregate_events(["group", "another_group"], "timestamp")
    assert df_wide.shape == (3, TIME_SLOTS)

    ax = df_long.cal.plot("timestamp")
    assert isinstance(ax, plt.Axes)

    df_long.cal.plot_across_column("timestamp", "group")


@pytest.fixture
def df_agg(df_long) -> pd.DataFrame:
    return (
        df_long.cal.timestamp_features("timestamp")
        .groupby(["group", "vocab"])
        .size()
        .rename("num_events")
        .to_frame()
    )


def test_agg_dataframe_extensions(df_agg) -> None:
    assert hasattr(df_agg, "cal")

    df_wide = df_agg.cal.widen("num_events")
    assert df_wide.shape == (3, TIME_SLOTS)

    with pytest.raises(ValueError):
        df_agg.reset_index(0).cal.widen("num_events")

    df_false_order = df_agg.reorder_levels([1, 0]).cal.widen("num_events")
    assert isinstance(df_false_order, pd.DataFrame)
    assert df_false_order.sum().sum() == 0


@pytest.fixture
def df_wide() -> pd.DataFrame:
    nrows = 25
    data = np.ones((nrows, TIME_SLOTS))
    return pd.DataFrame(data, columns=FULL_VOCAB)


class MockModel:
    def __init__(self) -> None:
        self.n_components = 3

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.ones((len(X), self.n_components))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones((len(X), TIME_SLOTS))


def test_wide_dataframe_extensions(df_wide: pd.DataFrame) -> None:
    assert hasattr(df_wide, "cal")
    nrows = len(df_wide)
    df_dow = df_wide.cal.sum_over_vocab("dow")

    df_dow_answer = pd.DataFrame(
        np.ones((nrows, DAYS_IN_WEEK)) * HOURS_IN_DAY,
        index=df_wide.index,
        columns=[day_of_week_column_name_func(i) for i in range(DAYS_IN_WEEK)],
    )

    pd.testing.assert_frame_equal(df_dow, df_dow_answer)

    df_hour = df_wide.cal.sum_over_vocab("hour")

    df_hour_answer = pd.DataFrame(
        np.ones((nrows, HOURS_IN_DAY)) * DAYS_IN_WEEK,
        index=df_wide.index,
    )

    pd.testing.assert_frame_equal(df_hour, df_hour_answer)

    df_wide.head(1).cal.plot_by_row()

    mock_model = MockModel()
    assert df_wide.cal.transform(model=mock_model).shape == (
        nrows,
        mock_model.n_components,
    )
    assert df_wide.cal.predict(model=mock_model).shape == (nrows, TIME_SLOTS)

    for head_rows in [1, 3]:
        df_plot = df_wide.head(head_rows).copy()

        df_plot.cal.plot_profile_by_row(model=mock_model, include_components=False)
        df_plot.cal.plot_profile_by_row(model=mock_model, include_components=True)
        df_plot.cal.plot_raw_and_predicted_by_row(model=mock_model)
        df_plot.cal.plot_model_predictions_by_row(df_plot, model=mock_model)

    for next_hours in [1, 2, 3]:
        df_answer = pd.DataFrame(
            np.ones((nrows, TIME_SLOTS)) * (1 + next_hours),
            index=df_wide.index,
            columns=FULL_VOCAB,
        )

        pd.testing.assert_frame_equal(
            df_wide.cal.sum_next_hours(hours=next_hours), df_answer
        )
