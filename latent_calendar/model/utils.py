import pandas as pd

from latent_calendar.model.latent_calendar import LatentCalendar


def _fit_with_features(model: LatentCalendar) -> bool:
    """https://github.com/scikit-learn/scikit-learn/blob/8721245511de2f225ff5f9aa5f5fadce663cd4a3/sklearn/base.py#L478-L497"""
    return hasattr(model, "features_names_in_")


def transform_on_dataframe(df: pd.DataFrame, model: LatentCalendar) -> pd.DataFrame:
    """Small wrapper to transform on DataFrame and keep index."""
    if _fit_with_features(model):
        return model.transform(df)

    return pd.DataFrame(model.transform(df.to_numpy()), index=df.index)


def predict_on_dataframe(df: pd.DataFrame, model: LatentCalendar) -> pd.DataFrame:
    """Small wrapper to predict on DataFrame and keep same columns and index."""
    if _fit_with_features(model):
        return model.predict(df)

    return pd.DataFrame(
        model.predict(df.to_numpy()), columns=df.columns, index=df.index
    )
