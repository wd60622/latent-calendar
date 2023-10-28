---
comments: true
---
# Scikit-Learn Compatibility

The `LatentCalendar` class is `scikit-learn` compatible, so we can use it in a pipeline. 

```python
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class RemoveLowVolumeTimeSlots(BaseEstimator, TransformerMixin):
    def __init__(self, min_count: int = 10):
        self.min_count = min_count

    def fit(self, X: pd.DataFrame, y=None):
        column_counts = X.sum()
        self.columns_to_keep = column_counts[column_counts > self.min_count].index

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.loc[:, self.columns_to_keep]


pipeline = Pipeline(
    [
        ("remove_low_volume", RemoveLowVolumeTimeSlots(min_count=10)),
        ("latent_calendar", LatentCalendar(n_components=10, random_state=42)),
    ]
)

pipeline.fit(df_wide)
```