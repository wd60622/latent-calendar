"""Pandas extensions for `latent-calendar` and primary interface for the package.

Provides a `cal` accessor to `DataFrame` and `Series` instances for easy transformation and plotting after import of `latent_calendar`.

Functionality includes: 

- aggregation of events to wide format
- convolutions of wide formats
- making transformations and predictions with models
- plotting of events, predictions, and comparisons as calendars

Each `DataFrame` will be either at event level or an aggregated wide format. 

Methods that end in `row` or `by_row` will be for wide format DataFrames and will plot each row as a calendar. 

Examples: 
    Plotting an event level Series as a calendar 

    ```python
    import pandas as pd
    import latent_calendar

    dates = pd.date_range("2023-01-01", "2023-01-14", freq="H")
    ser = (
        pd.Series(dates)
        .sample(10, random_state=42)
    )

    ser.cal.plot()
    ```

    ![Series Calendar](./../images/series-calendar.png)

    Transform event level DataFrame to wide format and plot

    ```python
    from latent_calendar.datasets import load_online_transactions

    df = load_online_transactions()

    # (n_customer, n_timeslots)
    df_wide = (
        df
        .cal.aggregate_events("Customer ID", timestamp_col="InvoiceDate")
    )

    (
        df_wide
        .sample(n=12, random_state=42)
        .cal.plot_by_row(max_cols=4)
    )
    ```

    ![Customer Transactions](./../images/customer-transactions.png)

    Train a model and plot predictions 

    ```python 
    from latent_calendar import LatentCalendar

    model = LatentCalendar(n_components=5, random_state=42)
    model.fit(df_wide.to_numpy())

    (
        df_wide
        .head(2)
        .cal.plot_profile_by_row(model=model)
    )
    ```

    ![Profile By Row](./../images/profile-by-row.png)


"""
from typing import List, Optional, Union

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from latent_calendar.model.latent_calendar import LatentCalendar
from latent_calendar.model.utils import transform_on_dataframe, predict_on_dataframe
from latent_calendar.plot.colors import CMAP, ColorMap
from latent_calendar.plot.core import (
    plot_calendar_by_row,
    plot_profile_by_row,
    plot_dataframe_as_calendar,
    plot_series_as_calendar,
    plot_dataframe_grid_across_column,
    plot_model_predictions_by_row,
)
from latent_calendar.plot.core.calendar import TITLE_FUNC, CMAP_GENERATOR
from latent_calendar.plot.elements import DayLabeler, TimeLabeler, GridLines
from latent_calendar.plot.iterate import StartEndConfig

from latent_calendar.segments.convolution import (
    sum_over_segments,
    sum_over_vocab,
    sum_next_hours,
)
from latent_calendar.transformers import create_raw_to_vocab_transformer


@pd.api.extensions.register_series_accessor("cal")
class SeriesAccessor:
    """Series accessor for latent_calendar accessed through `cal` attribute of Series."""

    def __init__(self, pandas_obj: pd.Series):
        self._obj = pandas_obj

    def plot(
        self,
        *,
        duration: int = 5,
        alpha: float = None,
        cmap=None,
        day_labeler: DayLabeler = DayLabeler(),
        time_labeler: TimeLabeler = TimeLabeler(),
        grid_lines: GridLines = GridLines(),
        monday_start: bool = True,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Plot Series of timestamps as a calendar.

        Args:
            duration: duration of each event in minutes
            alpha: alpha value for the color
            cmap: function that maps floats to string colors
            day_labeler: DayLabeler instance
            time_labeler: TimeLabeler instance
            grid_lines: GridLines instance
            monday_start: whether to start the week on Monday or Sunday
            ax: matplotlib axis to plot on

        Returns:
            Modified matplotlib axis

        """
        tmp_name = "tmp_name"
        config = StartEndConfig(start=tmp_name, end=None, minutes=duration)

        return plot_dataframe_as_calendar(
            self._obj.rename(tmp_name).to_frame(),
            config=config,
            alpha=alpha,
            cmap=cmap,
            monday_start=monday_start,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
            grid_lines=grid_lines,
            ax=ax,
        )

    def plot_row(
        self,
        *,
        alpha: float = None,
        cmap=None,
        day_labeler: DayLabeler = DayLabeler(),
        time_labeler: TimeLabeler = TimeLabeler(),
        grid_lines: GridLines = GridLines(),
        monday_start: bool = True,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Plot Series of timestamps as a calendar.

        Args:
            alpha: alpha value for the color
            cmap: function that maps floats to string colors
            monday_start: whether to start the week on Monday or Sunday
            ax: matplotlib axis to plot on

        Returns:
            Modified matplotlib axis

        """
        return plot_series_as_calendar(
            self._obj,
            alpha=alpha,
            cmap=cmap,
            ax=ax,
            monday_start=monday_start,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
            grid_lines=grid_lines,
        )


@pd.api.extensions.register_dataframe_accessor("cal")
class DataFrameAccessor:
    """DataFrame accessor for latent_calendar accessed through `cal` attribute of DataFrames."""

    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    def normalize(self, kind: str) -> pd.DataFrame:
        """Row-wise operations on DataFrame.

        Args:
            kind: one of ['max', 'probs', 'even_rate']

        Returns:
            DataFrame with row-wise operations applied

        """
        if kind == "max":
            return self._obj.div(self._obj.max(axis=1), axis=0)
        elif kind == "probs":
            return self._obj.div(self._obj.sum(axis=1), axis=0)
        elif kind == "even_rate":
            value = self._obj.shape[1]
            return self._obj.div(value)

        raise ValueError(f"kind must be one of ['max', 'probs'], got {kind}")

    def aggregate_events(
        self,
        by: Union[str, List[str]],
        timestamp_col: str,
    ) -> pd.DataFrame:
        """Transform DataFrame to wide format with groups as index.

        Wrapper around `create_raw_to_vocab_transformer` to transform to wide format.

        Args:
            by: column(s) to use as index
            timestamp_col: column to use as timestamp

        Returns:
            DataFrame in wide format

        """
        if not isinstance(by, list):
            id_col = by
            additional_groups = None
        else:
            id_col, *additional_groups = by

        transformer = create_raw_to_vocab_transformer(
            id_col=id_col,
            timestamp_col=timestamp_col,
            additional_groups=additional_groups,
        )
        return transformer.fit_transform(self._obj)

    def sum_over_vocab(self, aggregation: str = "dow") -> pd.DataFrame:
        """Sum the wide format to day of week or hour of day.

        Args:
            aggregation: one of ['dow', 'hour']

        Returns:
            DataFrame with summed values

        Examples:
            Sum to day of week

            ```python
            df_dow = df_wide.cal.sum_over_vocab(aggregation='dow')
            ```

        """
        return sum_over_vocab(self._obj, aggregation=aggregation)

    def sum_next_hours(self, hours: int) -> pd.DataFrame:
        """Sum the wide format over next hours.

        Args:
            hours: number of hours to sum over

        Returns:
            DataFrame with summed values

        """
        return sum_next_hours(self._obj, hours=hours)

    def sum_over_segments(self, df_segments: pd.DataFrame) -> pd.DataFrame:
        """Sum the wide format over user defined segments.

        Args:
            df_segments: DataFrame in wide format with segments as index

        Returns:
            DataFrame with columns as the segments and summed values

        """
        return sum_over_segments(self._obj, df_segments=df_segments)

    def transform(self, *, model: LatentCalendar) -> pd.DataFrame:
        """Transform DataFrame with model.

        Applies the dimensionality reduction to each row of the DataFrame.

        Args:
            model: model to use for transformation

        Returns:
            DataFrame with transformed values

        """
        return transform_on_dataframe(self._obj, model=model)

    def predict(self, *, model: LatentCalendar) -> pd.DataFrame:
        """Predict DataFrame with model.

        Args:
            model: model to use for prediction

        Returns:
            DataFrame with predicted values (wide format)

        """
        return predict_on_dataframe(self._obj, model=model)

    def plot(
        self,
        start_col: str,
        *,
        end_col: Optional[str] = None,
        duration: Optional[int] = None,
        alpha: float = None,
        cmap=None,
        day_labeler: DayLabeler = DayLabeler(),
        time_labeler: TimeLabeler = TimeLabeler(),
        grid_lines: GridLines = GridLines(),
        monday_start: bool = True,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Plot DataFrame of timestamps as a calendar.

        Args:
            start_col: column with start timestamp
            end_col: column with end timestamp
            duration: length of event in minutes. Alternative to end_col
            alpha: alpha value for the color
            cmap: function that maps floats to string colors
            monday_start: whether to start the week on Monday or Sunday
            ax: optional matplotlib axis to plot on

        Returns:
            Modified matplotlib axis

        """
        config = StartEndConfig(start=start_col, end=end_col, minutes=duration)

        return plot_dataframe_as_calendar(
            self._obj,
            config=config,
            alpha=alpha,
            cmap=cmap,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
            grid_lines=grid_lines,
            monday_start=monday_start,
            ax=ax,
        )

    def plot_across_column(
        self,
        start_col: str,
        grid_col: str,
        *,
        end_col: Optional[str] = None,
        duration: Optional[int] = None,
        day_labeler: DayLabeler = DayLabeler(),
        time_labeler: TimeLabeler = TimeLabeler(),
        grid_lines: GridLines = GridLines(),
        max_cols: int = 3,
        alpha: float = None,
    ) -> None:
        """Plot DataFrame of timestamps as a calendar as grid across column values.

        NA values are excluded

        Args:
            start_col: column with start timestamp
            grid_col: column of values to use as grid
            end_col: column with end timestamp
            duration: length of event in minutes. Alternative to end_col
            max_cols: max number of columns per row
            alpha: alpha value for the color

        Returns:
            None

        """
        config = StartEndConfig(start=start_col, end=end_col, minutes=duration)

        plot_dataframe_grid_across_column(
            self._obj,
            grid_col=grid_col,
            config=config,
            max_cols=max_cols,
            alpha=alpha,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
            grid_lines=grid_lines,
        )

    def plot_by_row(
        self,
        *,
        max_cols: int = 3,
        title_func: Optional[TITLE_FUNC] = None,
        cmaps: Optional[Union[CMAP, ColorMap, CMAP_GENERATOR]] = None,
        day_labeler: DayLabeler = DayLabeler(),
        time_labeler: TimeLabeler = TimeLabeler(),
        grid_lines: GridLines = GridLines(),
        monday_start: bool = True,
    ) -> None:
        """Plot each row of the DataFrame as a calendar plot. Data must have been transformed to wide format first.

        Wrapper around `latent_calendar.plot.plot_calendar_by_row`.

        Args:
            max_cols: max number of columns per row of grid
            title_func: function to generate title for each row
            day_labeler: function to generate day labels
            time_labeler: function to generate time labels
            cmaps: optional generator of colormaps
            grid_lines: optional grid lines
            monday_start: whether to start the week on Monday or Sunday

        Returns:
            None

        """
        return plot_calendar_by_row(
            self._obj,
            max_cols=max_cols,
            title_func=title_func,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
            cmaps=cmaps,
            grid_lines=grid_lines,
            monday_start=monday_start,
        )

    def plot_profile_by_row(
        self,
        *,
        model: LatentCalendar,
        index_func=lambda idx: idx,
        include_components: bool = True,
        day_labeler: DayLabeler = DayLabeler(),
        time_labeler: TimeLabeler = TimeLabeler(),
    ) -> np.ndarray:
        """Plot each row of the DataFrame as a profile plot. Data must have been transformed to wide format first.

        Args:
            model: model to use for prediction and transform
            index_func: function to generate title for each row
            include_components: whether to include components in the plot
            day_labeler: DayLabeler instance to use for day labels
            time_labeler: TimeLabeler instance to use for time labels

        Returns:
            grid of axes

        """
        return plot_profile_by_row(
            self._obj,
            model=model,
            index_func=index_func,
            include_components=include_components,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
        )

    def plot_raw_and_predicted_by_row(
        self,
        *,
        model: LatentCalendar,
        index_func=lambda idx: idx,
        day_labeler: DayLabeler = DayLabeler(),
        time_labeler: TimeLabeler = TimeLabeler(),
    ) -> np.ndarray:
        """Plot raw and predicted values for a model. Data must have been transformed to wide format first.

        Args:
            model: model to use for prediction
            index_func: function to generate title for each row
            day_labeler: DayLabeler instance to use for day labels
            time_labeler: TimeLabeler instance to use for time labels

        Returns:
            grid of axes

        """
        return plot_profile_by_row(
            self._obj,
            model=model,
            index_func=index_func,
            include_components=False,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
        )

    def plot_model_predictions_by_row(
        self,
        df_holdout: pd.DataFrame,
        *,
        model: LatentCalendar,
        index_func=lambda idx: idx,
        divergent: bool = True,
        day_labeler: DayLabeler = DayLabeler(),
        time_labeler: TimeLabeler = TimeLabeler(),
    ) -> np.ndarray:
        """Plot model predictions for each row of the DataFrame. Data must have been transformed to wide format first.

        Args:
            df_holdout: holdout DataFrame for comparison
            model: model to use for prediction
            index_func: function to generate title for each row
            divergent: whether to use divergent colormap
            day_labeler: DayLabeler instance to use for day labels
            time_labeler: TimeLabeler instance to use for time labels

        Returns:
            grid of axes

        """
        return plot_model_predictions_by_row(
            self._obj,
            df_holdout=df_holdout,
            model=model,
            index_func=index_func,
            divergent=divergent,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
        )
