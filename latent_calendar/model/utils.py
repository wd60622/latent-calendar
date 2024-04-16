import pandas as pd

from latent_calendar.model.latent_calendar import LatentCalendar


def transform_on_dataframe(df: pd.DataFrame, model: LatentCalendar) -> pd.DataFrame:
    """Small wrapper to transform on DataFrame and keep index."""
    return pd.DataFrame(model.transform(df.to_numpy()), index=df.index)


def predict_on_dataframe(df: pd.DataFrame, model: LatentCalendar) -> pd.DataFrame:
    """Small wrapper to predict on DataFrame and keep same columns and index."""
    return pd.DataFrame(
        model.predict(df.to_numpy()), columns=df.columns, index=df.index
    )
