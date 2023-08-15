"""The specific elements on the calendar plot. 

Includes x-axis, y-axis, and their settings, as well as the calendar events. 

"""
import calendar
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from latent_calendar.const import HOURS_IN_DAY, DAYS_IN_WEEK
from latent_calendar.plot.iterate import CalendarData
from latent_calendar.plot.colors import CMAP
from latent_calendar.vocab import HourFormatter, get_day_hour


@dataclass
class DisplaySettings:
    """Small wrapper to hold the display settings in the plots."""

    x: bool = True
    y: bool = True


@dataclass
class TimeLabeler:
    """This is the y-axis and all its settings in the plot.

    Also possible to be an x-axis for other plots as well.

    """

    hour_formatter: HourFormatter = HourFormatter()
    start: int = 0
    stride: int = 2
    display: bool = True
    rotation: Optional[float] = 0

    def get_hours(self) -> Tuple[List[int], List[str]]:
        return range(HOURS_IN_DAY + 1)[:: self.stride]

    def create_labels(self, ax: plt.Axes, axis: str = "y") -> None:
        """Create the hour labels on the plot ax."""
        if axis not in {"x", "y"}:
            raise ValueError("Only supported for the x and y.")

        hours = self.get_hours()
        hour_name_func = self.label if self.display else self.empty_label
        hour_names = hour_name_func(hours)

        getattr(ax, f"set_{axis}ticks")(hours, hour_names, rotation=self.rotation)
        getattr(ax, f"set_{axis}lim")(0, HOURS_IN_DAY)

        if axis == "y":
            ax.invert_yaxis()

    def label(self, hrs: List[int]) -> List[str]:
        return [self.hour_formatter(hr) for hr in hrs]

    def empty_label(self, hrs: List[int]) -> List[str]:
        return ["" for _ in range(len(hrs))]


def create_default_days():
    return [calendar.day_abbr[value] for value in range(DAYS_IN_WEEK)]


@dataclass
class DayLabeler:
    """This is typically the x-axis and all its settings in the plot."""

    day_start: int = 0
    days_of_week: List[str] = field(default_factory=create_default_days)
    rotation: Optional[float] = 45
    display: bool = True
    monday_start: bool = True

    def __post_init__(self) -> None:
        if self.day_start not in range(DAYS_IN_WEEK):
            msg = f"'day_start' value must be 0: Monday or 6: Sunday"
            raise ValueError(msg)

    @property
    def day_labels(self) -> List[str]:
        """What is added to the plot. If this is display, empty ticks."""
        if not self.display:
            return ["" for _ in range(DAYS_IN_WEEK)]

        return self.days_of_week[self.day_start :] + self.days_of_week[: self.day_start]

    def create_labels(self, ax: plt.Axes, axis: str = "x") -> None:
        """Create the labels for the plot."""
        getattr(ax, f"set_{axis}lim")(0, DAYS_IN_WEEK)
        getattr(ax, f"set_{axis}ticks")(
            [i + 0.5 for i in range(DAYS_IN_WEEK)],
            self.day_labels,
            rotation=self.rotation,
        )

        if axis == "y":
            ax.invert_yaxis()


@dataclass
class PlotAxes:
    """This configures the x and y axis in the plots."""

    day_labeler: DayLabeler = field(default_factory=DayLabeler)
    time_labeler: TimeLabeler = field(default_factory=TimeLabeler)

    @classmethod
    def axes_to_display(cls, x: bool = True, y: bool = True) -> "PlotAxes":
        display_settings = DisplaySettings(x=x, y=y)

        plot_axes = cls()
        plot_axes.update_display_settings(display_settings=display_settings)

        return plot_axes

    def configure_axis(self, ax: plt.Axes) -> None:
        self.day_labeler.create_labels(ax=ax)
        self.time_labeler.create_labels(ax=ax)

    def update_display_settings(self, display_settings: DisplaySettings) -> None:
        self.day_labeler.display = display_settings.x
        self.time_labeler.display = display_settings.y

    def update_start(self, monday_start: bool) -> None:
        self.day_labeler.day_start = 0 if monday_start else 6


@dataclass
class GridLines:
    """Grid lines between the calendar for the plot."""

    dow: bool = False
    hour: bool = False
    color: str = "black"
    linestyle: str = "--"
    alpha: float = 0.2

    def configure_grid(self, ax: plt.Axes) -> None:
        if self.dow:
            for dow in range(DAYS_IN_WEEK):
                ax.axvline(
                    x=dow + 1,
                    color=self.color,
                    linestyle=self.linestyle,
                    alpha=self.alpha,
                )

        if self.hour:
            for hour in range(HOURS_IN_DAY):
                ax.axhline(
                    y=hour + 1,
                    color=self.color,
                    linestyle=self.linestyle,
                    alpha=self.alpha,
                )


DEFAULT_LW = 0.1


@dataclass
class CalendarEvent:
    """Something on the calendar.

    Plots rectangles on axis

    Examples:
        Plot event from calendar data

        >>> calendar_data = CalendarData(day=0, start=0, end=2.5)
        >>> event = CalendarEvent.from_calendar_data(calendar_data=calendar_data, cmap=...)
        >>> event.plot_event(ax=ax)

        Plot a single calendar event from vocab

        >>> event = CalendarEvent.from_vocab("00 01")
        >>> event.plot_event(ax=ax)


    """

    day: int
    start: float
    end: float
    fillcolor: Optional[str] = None
    fill: bool = True
    alpha: Optional[float] = None
    lw: Optional[float] = None
    linestyle: Optional[str] = None

    @classmethod
    def from_calendar_data(
        cls, calendar_data: CalendarData, cmap: CMAP, alpha: Optional[float] = None
    ) -> "CalendarEvent":
        return cls(
            day=calendar_data.day,
            start=calendar_data.start,
            end=calendar_data.end,
            fillcolor=cmap(calendar_data.value),
            alpha=alpha,
            lw=calendar_data.lw,
        )

    @classmethod
    def from_vocab(
        cls,
        vocab: str,
        hours: float = 1.0,
        fillcolor: Optional[str] = None,
        fill: bool = False,
        alpha: Optional[float] = None,
        lw: float = 1.5,
        linestyle: Optional[str] = "dashed",
    ) -> "CalendarEvent":
        """Constructor from vocab string in order to plot on an axis.

        TODO: Ability to have the number of days
        TODO: Ability for the number of days to have wrapping as well.

        Example:
            Plot on an axis

            >>> event = CalendarEvent.from_vocab("00 01")
            >>> event.plot_event(ax=ax)

            Plot a two and half hour window

            >>> event = CalendarEvent.from_vocab("00 01", hours=2.5)
            >>> event.plot_event(ax=ax)

        """
        day, hour = get_day_hour(vocab=vocab)

        return cls(
            day=day,
            start=hour,
            end=hour + hours,
            fillcolor=fillcolor,
            fill=fill,
            alpha=alpha,
            lw=lw,
            linestyle=linestyle,
        )

    @property
    def multiday_tour(self) -> bool:
        if self.end == HOURS_IN_DAY:
            return False

        return self.end % HOURS_IN_DAY < self.start

    def _cap_event_at_midnight(self) -> "CalendarEvent":
        self.end = min(HOURS_IN_DAY, self.end)

    def _create_next_day_event(self) -> "CalendarEvent":
        """In the case of tour going into the next day, this is the next item."""
        return CalendarEvent(
            day=(self.day + 1) % DAYS_IN_WEEK,
            start=0,
            end=self.end % HOURS_IN_DAY,
            fillcolor=self.fillcolor,
            fill=self.fill,
            alpha=self.alpha,
            lw=self.lw,
            linestyle=self.linestyle,
        )

    def separate_events(self) -> List["CalendarEvent"]:
        """Return list of events that represent the one event across different days."""
        events = [replace(self)]

        if self.multiday_tour:
            events.append(self._create_next_day_event())
            # Cap the initial rectangle at 24 hours
            events[0]._cap_event_at_midnight()

        return events

    def _create_matplotlib_rectangle(
        self, monday_start: bool, **kwargs
    ) -> plt.Rectangle:
        """Create a rectangle matplotlib instance from the event."""
        height = self.end - self.start
        assert (
            height > 0.0
        ), f"The rectangle doesn't have positive height. Hour start {self.start} > Hour end {self.end}"

        x = self.day if monday_start else (self.day + 1) % DAYS_IN_WEEK
        rect_kwargs = {
            "xy": [x, self.start],
            "width": 1,
            "height": height,
            "edgecolor": "black",
            "lw": self.lw or DEFAULT_LW,
            "fill": self.fill,
            "linestyle": self.linestyle,
        }
        if self.fillcolor is not None:
            rect_kwargs["facecolor"] = self.fillcolor

        if self.alpha is not None:
            rect_kwargs["alpha"] = self.alpha

        return plt.Rectangle(**rect_kwargs, **kwargs)

    def plot_event(self, ax: plt.Axes, monday_start: bool = True, **kwargs) -> None:
        """Put the CalendarEvent instance onto an axis.


        Options for kwargs here:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html

        Args:
            ax: Axis to plot on
            monday_start: Whether to start the week on Monday or Sunday.
            kwargs: Addtional kwargs for the Patch instances or to override.

        """
        separated_events = self.separate_events()
        for event in separated_events:
            rectangle = event._create_matplotlib_rectangle(
                monday_start=monday_start, **kwargs
            )
            ax.add_patch(rectangle)
