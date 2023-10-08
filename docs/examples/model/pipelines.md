---
comments: true
---

The `LatentCalendar` class is `sklearn` compatible, so we can use it in a pipeline. 

```python
from sklearn.base import BaseEstimator, TransformerMixin


class RemoveLowVolumnTimeSlots(BaseEstimator, TransformerMixin):
    def __init__(self, min_count: int = 10):
        self.min_count = min_count

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.loc[:, X.sum() > self.min_count]
```