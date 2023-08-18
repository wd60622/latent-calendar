```python 
import matplotlib.pyplot as plt

from latent_calendar.datasets import load_online_transactions

df = load_online_transactions()
df.head()
```

```text
        StockCode                          Description  Quantity         InvoiceDate  Price  Customer ID         Country
Invoice                                                                                                                 
489434      85048  15CM CHRISTMAS GLASS BALL 20 LIGHTS        12 2009-12-01 07:45:00   6.95      13085.0  United Kingdom
489434     79323P                   PINK CHERRY LIGHTS        12 2009-12-01 07:45:00   6.75      13085.0  United Kingdom
489434     79323W                  WHITE CHERRY LIGHTS        12 2009-12-01 07:45:00   6.75      13085.0  United Kingdom
489434      22041         RECORD FRAME 7" SINGLE SIZE         48 2009-12-01 07:45:00   2.10      13085.0  United Kingdom
489434      21232       STRAWBERRY CERAMIC TRINKET BOX        24 2009-12-01 07:45:00   1.25      13085.0  United Kingdom
```

By default, a new `cal` attribute will be added to DataFrames given access to module functionality.

```python
df_wide = df.cal.aggregate_events("Country", "InvoiceDate")

df_wide.head()
```

```text
vocab      00 00  00 01  00 02  00 03  00 04  00 05  00 06  00 07  00 08  ...  06 15  06 16  06 17  06 18  06 19  06 20  06 21  06 22  06 23
Country                                                                   ...                                                               
Australia      0      0      0      0      0      0      0      0      0  ...      0      0      0      0      0      0      0      0      0
Austria        0      0      0      0      0      0      0      0      9  ...      0      0      0      0      0      0      0      0      0
Bahrain        0      0      0      0      0      0      0      0      0  ...      0      0      0      0      0      0      0      0      0
Belgium        0      0      0      0      0      0      0      0     15  ...    131      0      0      0      0      0      0      0      0
Bermuda        0      0      0      0      0      0      0      0      0  ...      0      0      0      0      0      0      0      0      0

[5 rows x 168 columns]
```

We can clearly see the weekly hours of operation for these countries. Even though these are online transactions, these hours affect the transaction times.

The slight shift in hours for the UK might be the difference in time zones between the UK and the other countries. Maybe it could be difference in buying patterns of these populations. Not sure but this visual gives us a good glance at the data and starting point to ask questions

```python 
countries = ["United Kingdom", "Germany", "France"]

(
    df_wide
    .loc[countries]
    .cal.normalize("max")
    .cal.plot_by_row()
)
fig = plt.gcf()
fig.suptitle("Store Transactions by Country")
plt.show()
```

![Store Transactions by Country](./../../images/store-transactions.png)