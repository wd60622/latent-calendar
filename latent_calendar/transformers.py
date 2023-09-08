"""scikit-learn transformers for the data.

```python 
from latent_calendar.datasets import load_online_transactions

df = load_online_transactions()

transformers = create_raw_to_vocab_transformer(id_col="Customer ID", timestamp_col="InvoiceDate")

df_wide = transformers.fit_transform(df)
```


"""
from typing import List, Optional, Union
from datetime import datetime

import pandas as pd
from pandas.core.indexes.accessors import DatetimeProperties

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from latent_calendar.const import (
    FULL_VOCAB,
    HOURS_IN_DAY,
    MINUTES_IN_DAY,
    SECONDS_IN_DAY,
    MICROSECONDS_IN_DAY,
)


def prop_into_day(dt: Union[datetime, DatetimeProperties]) -> Union[float, pd.Series]:
    """Returns the proportion into the day from datetime like object.

    0.0 is midnight and 1.0 is midnight again.

    Args:
        dt: datetime like object

    Returns:
        numeric value(s) between 0.0 and 1.0

    """
    prop_hour = dt.hour / HOURS_IN_DAY
    prop_minute = dt.minute / MINUTES_IN_DAY
    prop_second = dt.second / SECONDS_IN_DAY
    prop_microsecond = dt.microsecond / MICROSECONDS_IN_DAY

    return prop_hour + prop_minute + prop_second + prop_microsecond


class CalandarTimestampFeatures(BaseEstimator, TransformerMixin):
    """Day of week and prop into day columns creation."""

    def __init__(
        self,
        timestamp_col: str,
    ) -> None:
        self.timestamp_col = timestamp_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Create 2 new columns."""
        if not hasattr(X[self.timestamp_col], "dt"):
            raise RuntimeError(
                f"Column {self.timestamp_col!r} is not a datetime column. Use df[{self.timestamp_col!r}] = pd.to_datetime(df[{self.timestamp_col!r}]) first."
            )

        X = X.copy()

        X["prop_into_day_start"] = prop_into_day(X[self.timestamp_col].dt)
        X["day_of_week"] = X[self.timestamp_col].dt.dayofweek

        X["hour"] = X["prop_into_day_start"] * 24

        tmp_columns = ["prop_into_day_start"]
        self.created_columns = ["day_of_week", "hour"]

        X = X.drop(columns=tmp_columns)
        self.columns = list(X.columns)

        return X

    def get_feature_names_out(self, input_features=None):
        return self.columns.extend(self.created_columns)


class HourDiscretizer(BaseEstimator, TransformerMixin):
    """Discretize the hour column.

    Args:
        col: The name of the column to discretize.
        minutes: The number of minutes to discretize by.

    """

    def __init__(self, col: str = "hour", minutes: int = 60) -> None:
        self.col = col
        self.minutes = minutes

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        divisor = 1 if self.minutes == 60 else self.minutes / 60
        X[self.col] = (X[self.col] // divisor) * divisor

        if self.minutes % 60 == 0:
            X[self.col] = X[self.col].astype(int)

        self.columns = list(X.columns)

        return X

    def get_feature_names_out(self, input_features=None):
        return self.columns


class VocabTransformer(BaseEstimator, TransformerMixin):
    """Create a vocab column from the day of week and hour columns."""

    def __init__(
        self, day_of_week_col: str = "day_of_week", hour_col: str = "hour"
    ) -> None:
        self.day_of_week_col = day_of_week_col
        self.hour_col = hour_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X["vocab"] = (
            X[self.day_of_week_col]
            .astype(str)
            .str.zfill(2)
            .str.cat(X[self.hour_col].astype(str).str.zfill(2), sep=" ")
        )

        self.columns = list(X.columns)

        return X

    def get_feature_names_out(self, input_features=None):
        return self.columns


def create_timestamp_feature_pipeline(
    timestamp_col: str,
    create_vocab: bool = True,
) -> Pipeline:
    """Create a pipeline that creates features from the timestamp column.

    Args:
        timestamp_col: The name of the timestamp column.

    Returns:
        A pipeline that creates features from the timestamp column.

    Example:
        Create features for the online transactions dataset.

        ```python
        from latent_calendar.datasets import load_online_transactions

        df = load_online_transactions()

        transformers = create_timestamp_feature_pipeline(timestamp_col="InvoiceDate")

        df_features = transformers.fit_transform(df)
        ```

    """
    vocab_col = "hour"
    transformers = [
        (
            "timestamp_features",
            CalandarTimestampFeatures(timestamp_col=timestamp_col),
        ),
        ("binning", HourDiscretizer(col=vocab_col)),
    ]

    if create_vocab:
        transformers.append(
            ("vocab_creation", VocabTransformer(hour_col=vocab_col)),
        )

    return Pipeline(
        transformers,
    ).set_output(transform="pandas")


class VocabAggregation(BaseEstimator, TransformerMixin):
    """NOTE: The index of the grouping stays."""

    def __init__(self, groups: List[str], cols: Optional[List[str]] = None) -> None:
        self.groups = groups
        self.cols = cols

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        stats = {}
        if self.cols is not None:
            stats.update({col: (col, "sum") for col in self.cols})

        df_agg = (
            X.assign(num_events=1)
            .groupby(self.groups)
            .agg(num_events=("num_events", "sum"), **stats)
        )
        self.columns = list(df_agg.columns)

        return df_agg

    def get_feature_names_out(self, input_features=None):
        return self.columns


class LongToWide(BaseEstimator, TransformerMixin):
    def __init__(self, col: str = "num_events", as_int: bool = True) -> None:
        self.col = col
        self.as_int = as_int

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Unstack the assumed last index as vocab column."""
        X_T = X.loc[:, self.col].unstack().T
        X_T.index = X_T.index.get_level_values(-1)
        X_T = X_T.reindex(FULL_VOCAB)
        X_res = X_T.T.fillna(value=0)
        if self.as_int:
            X_res = X_res.astype(int)

        return X_res

    def get_feature_names_out(self, input_features=None):
        return FULL_VOCAB


class RawToVocab(BaseEstimator, TransformerMixin):
    """Transformer timestamp level data into id level data with vocab columns."""

    def __init__(
        self,
        id_col: str,
        timestamp_col: str,
        additional_groups: Optional[List[str]] = None,
        cols: Optional[List[str]] = None,
    ) -> None:
        self.id_col = id_col
        self.timestamp_col = timestamp_col
        self.additional_groups = additional_groups
        self.cols = cols

    def fit(self, X: pd.DataFrame, y=None):
        # New features at same index level
        self.features = create_timestamp_feature_pipeline(
            self.timestamp_col,
        )

        groups = [self.id_col]
        if self.additional_groups is not None:
            if not isinstance(self.additional_groups, list):
                raise ValueError(
                    f"additional_groups should be list not {type(self.additional_groups)}"
                )

            groups.extend(self.additional_groups)
        groups.append("vocab")

        # Reaggregation
        self.aggregation = VocabAggregation(groups=groups, cols=self.cols)
        # Unstacking
        self.widden = LongToWide(col="num_events")
        # Since nothing needs to be "fit"
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_trans = self.features.transform(X)

        X_agg = self.aggregation.transform(X_trans)
        return self.widden.transform(X_agg)


def create_raw_to_vocab_transformer(
    id_col: str,
    timestamp_col: str,
    additional_groups: Optional[List[str]] = None,
) -> RawToVocab:
    """Wrapper to create the transformer from the configuration options."""
    return RawToVocab(
        id_col=id_col,
        timestamp_col=timestamp_col,
        additional_groups=additional_groups,
    )
