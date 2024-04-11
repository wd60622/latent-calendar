import pytest

import numpy as np

from latent_calendar.generate import sample_from_lda


@pytest.mark.skip(reason="scipy issue from arviz")
def test_sample_from_lda() -> None:
    n_samples = np.array([10, 20, 30])

    n_rows = len(n_samples)

    components_prior = np.array([0.5, 0.5])
    components_time_slots_prior = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    df_components, df_wide = sample_from_lda(
        components_prior=components_prior,
        components_time_slots_prior=components_time_slots_prior,
        n_samples=n_samples,
        random_state=42,
    )

    assert df_components.shape == (n_rows, 2)
    assert df_wide.shape == (n_rows, 3)
