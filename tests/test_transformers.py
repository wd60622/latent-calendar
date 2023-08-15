import pytest

import pandas as pd

from latent_calendar.transformers import (
    prop_into_day,
    CalandarTimestampFeatures,
    create_timestamp_feature_pipeline,
    create_raw_to_vocab_transformer,
)


@pytest.fixture
def sample_timestamp_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "datetime": pd.to_datetime(
                [
                    "2021-01-01 12:00",
                    "2021-01-01 13:55",
                    "2021-01-01 14:05",
                    "2021-01-01 14:55",
                ]
            ),
            "another_grouping": [1, 1, 1, 2],
        },
        index=pd.Index(["first", "second", "third", "fourth"]),
    )


@pytest.mark.parametrize(
    "date", ["2023-07-01", "2023-01-01", "2020-01-01", "1970-01-01"]
)
def test_prop_into_day_series(date) -> None:
    times = ["00:00", "01:00", "12:00", "23:59"]
    answers = [0.0, 1 / 24, 0.5, 0.9993]
    dates = pd.Series(pd.to_datetime([f"{date} {time}" for time in times]))
    results = prop_into_day(dates.dt)
    print(results)
    answer = pd.Series(answers)

    pd.testing.assert_series_equal(results, answer, atol=0.001)


@pytest.mark.parametrize("pandas_output", [True, False])
def test_calendar_timestamp_features(
    sample_timestamp_df: pd.DataFrame, pandas_output: bool
) -> None:
    timestamp_features = CalandarTimestampFeatures(
        timestamp_col="datetime",
    )
    if pandas_output:
        timestamp_features.set_output(transform="pandas")

    df_result = timestamp_features.fit_transform(sample_timestamp_df)

    assert isinstance(df_result, pd.DataFrame)
    cols_to_check = ["hour", "day_of_week"]
    for col in cols_to_check:
        assert col in df_result.columns
        assert col not in sample_timestamp_df.columns

    assert len(df_result.columns) == len(sample_timestamp_df.columns) + len(
        cols_to_check
    )


def test_timestamp_features(sample_timestamp_df: pd.DataFrame) -> None:
    pipe = create_timestamp_feature_pipeline(timestamp_col="datetime")

    df_result = pipe.fit_transform(sample_timestamp_df.copy())

    assert isinstance(df_result, pd.DataFrame)
    assert len(df_result) == len(sample_timestamp_df)


def test_raw_to_vocab(sample_timestamp_df) -> None:
    pipeline = create_raw_to_vocab_transformer(id_col="id", timestamp_col="datetime")

    df_result = pipeline.fit_transform(sample_timestamp_df.copy())

    assert isinstance(df_result, pd.DataFrame)
    assert isinstance(df_result.index, pd.Index)
    assert len(df_result) == len(sample_timestamp_df["id"].unique())


def test_raw_to_vocab_with_groups(sample_timestamp_df) -> None:
    pipeline = create_raw_to_vocab_transformer(
        id_col="id", timestamp_col="datetime", additional_groups=["another_grouping"]
    )

    df_result = pipeline.fit_transform(sample_timestamp_df.copy())

    assert isinstance(df_result, pd.DataFrame)
    assert isinstance(df_result.index, pd.MultiIndex)
