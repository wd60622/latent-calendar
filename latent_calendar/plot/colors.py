"""Handling the colors in the calendar plots.

This module provides some helper function and some defaults. However, they might 
not be the best of all purposes.

TODO: Get a better pallet overall :D
TODO: Be able to distinguish between counts, probability, and relative numbers

"""

from typing import Callable, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, Normalize
from matplotlib.cm import ScalarMappable

import numpy as np

from latent_calendar.const import EVEN_PROBABILITY

from latent_calendar.plot.config import CONFIG


CM = Callable[[float], Tuple[int, int, int, int]]
CMAP = Callable[[float], str]


class ColorMap(ScalarMappable):
    """This supports colorbar for a figure from matplotlib.

    TODO: Add into the selective plots
    TODO: Consider having the label or even a binning of the colorbar

    """

    def __init__(self, norm, cmap, default_cm: CM) -> None:
        cmap = cmap if cmap is not None else default_cm
        super().__init__(norm=norm, cmap=cmap)

    def __call__(self, x: float) -> str:
        return rgb2hex(self.cmap(self.norm(x)))

    def add_colorbar(self, ax=None) -> None:
        """Add the colorbar to axis or axes.

        Args:
            ax: single or np.ndarray of Axes

        """
        fig = plt.gcf()

        fig.colorbar(self, ax=ax, ticks=self.norm.ticks)


def create_cmap(
    max_value: float, min_value: float = 0.0, cm: Optional[CM] = None
) -> CMAP:
    """Create color map function

    Might be good for values from low to high.

    """
    norm = Normalize(vmin=min_value, vmax=max_value)
    norm.ticks = [norm.vmin, norm.vmax]
    return ColorMap(norm=norm, cmap=cm, default_cm=plt.cm.YlGn)


def create_diverge_cmap(
    center_value: float, range: float, cm: Optional[CM] = None
) -> CMAP:
    """Create color map function to emphasize a center value and deviation from that center.

    Might be good for values that are relative to some baseline

    """
    half_range = range / 2
    norm = Normalize(vmin=center_value - half_range, vmax=center_value + half_range)
    norm.ticks = [norm.vmin, 0, norm.vmax]
    return ColorMap(norm=norm, cmap=cm, default_cm=plt.cm.coolwarm)


def create_relative_cmap(range: float) -> CMAP:
    """Good for relative scales."""
    return create_diverge_cmap(center_value=1.0, range=range)


# Helper Functions
def create_default_cmap(value: float) -> CMAP:
    max_value = value * CONFIG.max_value_ratio

    return create_cmap(min_value=0.0, max_value=max_value)


def create_default_divergent_cmap() -> CMAP:
    return create_diverge_cmap(center_value=1.0, range=CONFIG.range)


def settle_data_and_cmap(data, divergent: bool) -> Tuple[np.ndarray, CMAP]:
    """Return a tuple of transformed data and cmap for displaying that data."""
    if divergent:
        # Comparing the values to random rate
        data = data / EVEN_PROBABILITY
        return data, create_default_divergent_cmap()

    return data, create_default_cmap(value=data.max())
