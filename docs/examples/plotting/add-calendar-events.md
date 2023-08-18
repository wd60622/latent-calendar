Can use the `CalendarEvent` class to add arbitrary events to calendar. 

The constructor takes a day of week, start time, and an end time or duration.

```python
import matplotlib.pyplot as plt

from latent_calendar.plot import plot_blank_calendar
from latent_calendar.plot.elements import CalendarEvent

ax = plot_blank_calendar()

event = CalendarEvent(
    day=0, start=12, duration=15
)
event.plot(ax=ax, label="15 minute event", linestyle="--", alpha=0.25)

event = CalendarEvent(
    day=0, start=23.5, end=24.5, 
)
event.plot(ax=ax, label="Two day event", facecolor="red", linestyle="dashed", lw=1.5)

event = CalendarEvent(
    day=5, start=11, end=17, 
)
event.plot(ax=ax, label="Friday event", facecolor="green", alpha=0.25)

ax.legend()
plt.show()
```


![Arbitrary Events](./../../images/arbitrary-events.png)