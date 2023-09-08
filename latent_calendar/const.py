import calendar
from itertools import product
from typing import Dict, List

import numpy as np

DAYS_IN_WEEK = 7

HOURS_IN_DAY = 24
MINUTES_IN_DAY = HOURS_IN_DAY * 60
SECONDS_IN_DAY = MINUTES_IN_DAY * 60
MICROSECONDS_IN_DAY = SECONDS_IN_DAY * 1000000

TIME_SLOTS = DAYS_IN_WEEK * HOURS_IN_DAY

EVEN_PROBABILITY = 1 / TIME_SLOTS

DAY_LOOKUP: Dict[str, int] = {
    **{calendar.day_abbr[value]: value for value in range(DAYS_IN_WEEK)},
    **{calendar.day_name[value]: value for value in range(DAYS_IN_WEEK)},
}
# Since pretty common
DAY_LOOKUP["Tues"] = 1
DAY_LOOKUP["Thurs"] = 3


def format_dow_hour(day_of_week: int, hour: int) -> str:
    return f"{day_of_week:02} {hour:02}"


def dicretized_hours(minutes: int) -> List[float]:
    step = minutes / 60
    hours = np.arange(0, HOURS_IN_DAY, step)
    if minutes % 60 == 0:
        return hours.astype(int).tolist()

    return hours.tolist()


def create_full_vocab(days_in_week: int, minutes: int) -> List[str]:
    return [
        format_dow_hour(day_of_week, hour)
        for day_of_week, hour in product(range(days_in_week), dicretized_hours(minutes))
    ]


FULL_VOCAB = create_full_vocab(days_in_week=DAYS_IN_WEEK, minutes=60)
