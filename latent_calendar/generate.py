"""Generate some fake data for various purposes."""

import numpy as np
import pandas as pd

from latent_calendar.const import FULL_VOCAB

try:
    import pymc as pm
    from pytensor.tensor import TensorVariable
except ImportError:

    class TensorVariable:
        pass

    class PyMC:
        def __getattr__(self, name):
            msg = (
                "PyMC is not installed."
                " Please install it directly or"
                " with extra installs: pip install 'latent-calendar[pymc]'"
            )
            raise ImportError(msg)

    pm = PyMC()


def wide_format_dataframe(
    n_rows: int,
    rate: float = 1.0,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Generate some data from Poisson distribution.

    Args:
        n_rows: number of rows to generate
        rate: rate parameter for Poisson distribution
        random_state: random state for reproducibility

    Returns:
        DataFrame with columns from FULL_VOCAB and n_rows rows

    """
    if random_state is not None:
        np.random.seed(random_state)

    data = np.random.poisson(lam=rate, size=(n_rows, len(FULL_VOCAB)))

    return pd.DataFrame(data, columns=FULL_VOCAB)


def define_single_user_samples(
    travel_style,
    time_slot_styles: TensorVariable,
    n_samples: int,
) -> tuple[TensorVariable, TensorVariable]:
    travel_style_user = pm.Categorical.dist(p=travel_style, shape=n_samples)
    time_slots = pm.Multinomial.dist(p=time_slot_styles[travel_style_user], n=1)

    return travel_style_user, time_slots


def sample_from_lda(
    components_prior: np.ndarray | TensorVariable,
    components_time_slots_prior: np.ndarray | TensorVariable,
    n_samples: np.ndarray,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample from LDA model.

    Args:
        components_prior: prior probability of each component (n_components, )
        components_time_slots_prior: prior for time slots (n_components, n_time_slots)
        n_samples: number of samples for each user (n_user, )
        random_state: random state for sampling

    Returns:
        probability DataFrame (n_user, n_components) and event count DataFrame with (n_user, n_time_slots) with each row summing up to `n`

    """
    rng = np.random.default_rng(random_state)

    user_travel_style_data = []
    user_time_slot_data = []

    travel_style = pm.Dirichlet.dist(components_prior)
    time_slot_styles = pm.Dirichlet.dist(components_time_slots_prior)

    for n in n_samples:
        _, user_time_slots = define_single_user_samples(
            travel_style, time_slot_styles, n_samples=int(n)
        )

        user_travel_style_samples, user_time_slot_samples = pm.draw(
            [travel_style, user_time_slots.sum(axis=0)], draws=1, random_seed=rng
        )

        user_travel_style_data.append(user_travel_style_samples)
        user_time_slot_data.append(user_time_slot_samples)

    df_user_travel_style = pd.DataFrame(user_travel_style_data)
    df_user_time_slots = pd.DataFrame(user_time_slot_data)

    return df_user_travel_style, df_user_time_slots
