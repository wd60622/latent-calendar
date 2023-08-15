Transform and visualize data on a weekly calendar with the [`cal` attribute of DataFrames](./modules/extensions.md).

```python
import matplotlib.pyplot as plt

from latent_calendar.datasets import load_chicago_bikes

df = load_chicago_bikes()
df_member_casual = df.cal.to_vocab("member_casual", "started_at")
```

```text

```


```python
(
    df_member_casual
    .cal.normalize("max")
    .cal.plot_by_row()
)
fig = plt.gcf()
fig.suptitle("Bike Rentals by Member Type")
plt.show()
```