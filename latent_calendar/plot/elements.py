"""The specific elements on the calendar plot. 

Includes x-axis, y-axis, and their settings, as well as the calendar events. 

"""
import calendar
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

from latent_calendar.const import HOURS_IN_DAY, DAYS_IN_WEEK
from latent_calendar.plot.iterate import CalendarData
from latent_calendar.vocab import HourFormatter, get_day_hour


@dataclass
class DisplaySettings:
    """Small wrapper to hold the display settings in the plots.

    Args:
        x: Whether to x axis the plot.
        y: Whether to y axis the plot.

    """

    x: bool = True
    y: bool = True


@dataclass
class TimeLabeler:
    """This is time of day and all its settings in the plot.

    This is typically the y-axis.

    Args:
        hour_formatter: The formatter for the hour labels.
        start: The hour to start the plot at.
        stride: The number of hours to skip between ticks.
        display: Whether to display the hour labels.
        rotation: The rotation of the hour labels.

    """

    hour_formatter: HourFormatter = field(default_factory=HourFormatter)
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
    """Day of the week axis.

    This is typically the x-axis.

    Args:
        day_start: The day to start the plot at.
        days_of_week: The names of the days of the week.
        rotation: The rotation of the day labels.
        display: Whether to display the day labels.

    """

    day_start: int = 0
    days_of_week: List[str] = field(default_factory=create_default_days)
    rotation: Optional[float] = 45
    display: bool = True

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


def configure_axis(day_labeler, time_labeler, ax: plt.Axes) -> None:
    day_labeler.create_labels(ax=ax)
    time_labeler.create_labels(ax=ax)


def update_display_settings(
    day_labeler, time_labeler, display_settings: DisplaySettings
) -> None:
    day_labeler.display = display_settings.x
    time_labeler.display = display_settings.y


def update_start(day_labeler, monday_start: bool) -> None:
    day_labeler.day_start = 0 if monday_start else 6


@dataclass
class GridLines:
    """Grid lines between the calendar for the plot.

    Args:
        dow: Whether to add day of week grid lines.
        hour: Whether to add hour grid lines.
        color: The color of the grid lines.
        linestyle: The style of the grid lines.
        alpha: The alpha of the grid lines.

    """

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


@dataclass
class CalendarEvent:
    """Something on the calendar.

    Plots rectangles on matplotlib axis.

    Args:
        day: The day of the week. 0 is Monday.
        start: The start hour of the event.
        end: The end hour of the event.
        duration: The duration of the event. Only used if end is None.

    Examples:
        Plot event from calendar data

        ```python
        calendar_data = CalendarData(day=0, start=0, end=2.5)
        event = CalendarEvent.from_calendar_data(calendar_data=calendar_data, cmap=...)
        event.plot(ax=ax)

        Plot a single calendar event from vocab

        ```python
        event = CalendarEvent.from_vocab("00 01")
        event.plot(ax=ax)
        ```

    """

    day: int
    start: float
    end: Optional[float] = None
    duration: Optional[float] = None

    def __post_init__(self) -> None:
        if self.end is None and self.duration is None:
            raise ValueError("Either end or duration must be provided")

        if self.end is not None and self.duration is not None:
            raise ValueError("Only one of end or duration can be provided")

        if self.end is None:
            self.end = self.start + (self.duration / 60)
            self.duration = None

    @classmethod
    def from_calendar_data(
        cls,
        calendar_data: CalendarData,
    ) -> "CalendarEvent":
        return cls(
            day=calendar_data.day,
            start=calendar_data.start,
            end=calendar_data.end,
        )

    @classmethod
    def from_vocab(
        cls,
        vocab: str,
        duration: float = 60.0,
    ) -> "CalendarEvent":
        """Constructor from vocab string in order to plot on an axis.

        Args:
            vocab: The vocab string.
            duration: The duration of the event.

        Example:
            Plot on an axis

            ```python
            event = CalendarEvent.from_vocab("00 01")
            event.plot(ax=ax)
            ```

            Plot a two and half hour window

            ```python
            event = CalendarEvent.from_vocab("00 01", hours=2.5)
            event.plot(ax=ax)
            ```

        """
        day, hour = get_day_hour(vocab=vocab)

        return cls(
            day=day,
            start=hour,
            end=hour + (duration / 60),
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
        )

    def separate_events(self) -> List["CalendarEvent"]:
        """Return list of events that represent the one event across different days.


        Examples:
            A single event that goes from 23:00 to 01:00 will be split into two events.

            ```python
            event = CalendarEvent(day=0, start=23, duration=2 * 60)

            events = event.separate_events()
            ```

        """
        events = [replace(self)]

        if self.multiday_tour:
            events.append(self._create_next_day_event())
            # Cap the initial rectangle at 24 hours
            events[0]._cap_event_at_midnight()

        return events

    def _create_matplotlib_rectangle(
        self, monday_start: bool, lw, fill: bool, linestyle, fillcolor, alpha, **kwargs
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
        }

        rect_kwargs["edgecolor"] = "black"
        rect_kwargs["lw"] = lw
        rect_kwargs["fill"] = fill
        rect_kwargs["linestyle"] = linestyle
        rect_kwargs["facecolor"] = fillcolor
        rect_kwargs["alpha"] = alpha

        rect_kwargs.update(kwargs)

        return plt.Rectangle(**rect_kwargs)

    def plot(
        self,
        ax: plt.Axes,
        monday_start: bool = True,
        lw: float = 0.1,
        fill: bool = True,
        linestyle=None,
        fillcolor=None,
        alpha=None,
        **kwargs,
    ) -> None:
        """Put the CalendarEvent instance onto an axis.

        Options for kwargs [here](https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html).

        Args:
            ax: Axis to plot on
            monday_start: Whether to start the week on Monday or Sunday.
            lw: The line width of the event.
            fill: Whether to fill the event.
            linestyle: The line style of the event.
            fillcolor: The color of the event.
            alpha: The alpha of the event.
            kwargs: Addtional kwargs for the Patch instances or to override.

        """
        separated_events = self.separate_events()
        for event in separated_events:
            rectangle = event._create_matplotlib_rectangle(
                monday_start=monday_start,
                lw=lw,
                fill=fill,
                linestyle=linestyle,
                fillcolor=fillcolor,
                alpha=alpha,
                **kwargs,
            )
            if "label" in kwargs:
                kwargs.pop("label")

            ax.add_patch(rectangle)
