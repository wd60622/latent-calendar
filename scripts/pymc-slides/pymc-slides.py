from functools import partial
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import pymc as pm
from pytensor import tensor as pt

from rich.console import Console

from lyft_bikes import read_historical_trips

from latent_calendar import datasets, LatentCalendar
from latent_calendar.plot import (
    plot_model_components,
    TimeLabeler,
    CalendarEvent,
    DayLabeler,
    create_default_divergent_cmap,
    plot_blank_calendar,
    GridLines,
)
from latent_calendar.vocab import make_human_readable

font = {"size": 15}

matplotlib.rc("font", **font)


console = Console()


HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR = HERE / "images"
IMAGE_DIR.mkdir(exist_ok=True)

# Monday
START_DATE = "2023-05-01"
# Sunday
END_DATE = "2023-07-30"
CITY = "Chicago"
TRIPS_FILE = DATA_DIR / "chicago-trips.pkl"


def read_chicago_trips():
    if not TRIPS_FILE.exists():
        df = read_historical_trips(start_date=START_DATE, end_date=END_DATE, city=CITY)
        for col in ["started_at", "ended_at"]:
            df[col] = pd.to_datetime(df[col])
        df.to_pickle(TRIPS_FILE)

    return pd.read_pickle(TRIPS_FILE)


spaced_print = partial(print, end="\n\n")


def print_bold(text):
    console.print(f"[bold]{text}[/bold]")


def savefig(
    name: str, fig=None, height: float = 12, width: float = 20, **kwargs
) -> None:
    if fig is None:
        fig = plt.gcf()

    fig.set_figwidth(width)
    fig.set_figheight(height)

    file = IMAGE_DIR / f"{name}.png"
    plt.savefig(file, **kwargs)
    plt.close()


def create_df_mock(n_events, vocab, time_slot: str) -> pd.DataFrame:
    ncols = len(vocab)
    df_mock = pd.DataFrame(
        np.zeros((len(n_events), ncols)), columns=vocab, index=n_events
    )

    df_mock.loc[:, time_slot] = n_events

    return df_mock


def plot_mock_data(time_slot: str, vocab, model: LatentCalendar) -> plt.Figure:
    n_events = np.array([0, 1, 2, 5, 7, 10, 20])
    df_mock = create_df_mock(n_events, vocab, time_slot)

    fig, axes = plt.subplots(ncols=4)
    fig.suptitle(f"Events on {make_human_readable(time_slot)}")

    df_mock_predictions = df_mock.cal.predict(model=model)
    event = CalendarEvent.from_vocab(time_slot)
    cmap = create_default_divergent_cmap()
    for idx, (ax, n_events_idx) in enumerate(zip(axes, [0, 1, -1])):
        n = n_events[n_events_idx]
        label = "events" if n != 1 else "event"
        ax.set(
            title=f"{n} {label}",
        )
        ncols = len(vocab)
        time_labeler = TimeLabeler(display=idx == 0)
        df_mock_predictions.mul(ncols).iloc[n_events_idx].cal.plot_row(
            ax=ax, time_labeler=time_labeler, cmap=cmap
        )
        event.plot(ax=ax, fill=False, edgecolor="black", lw=2, linestyle="--")

    ax = axes[-1]
    df_mock.cal.transform(model=model).plot(ax=ax)
    ax.set(
        title="Component probabilities",
        ylim=(0, 1),
        xlabel="# of events",
    )
    # Add a name to the legend
    ax.get_legend().set_title("Component")

    return fig


def main() -> None:
    pass


if __name__ == "__main__":
    df = read_chicago_trips()
    # Since shorter
    df.index.name = "ride_id"

    df["weekofyear"] = df["started_at"].dt.isocalendar().week
    df["after"] = df["weekofyear"].ge(27)

    ser = (
        df.dropna(subset=["start_station_name"])
        .set_index(["start_station_name"], append=True)
        .loc[:, "started_at"]
        .reorder_levels([1, 0])
        .sort_index()
    )

    stations = ["DuSable Lake Shore Dr & North Blvd", "Clark St & Schiller St"]
    stations = ["Canal St & Adams St", "Streeter Dr & Grand Ave"]
    print_bold("Series")
    spaced_print(ser.loc[stations].to_frame())

    print(
        len(df) / 1_000_000,
        "trips from",
        df["started_at"].min(),
        "to",
        df["started_at"].max(),
    )

    print_bold("Series with calendar features")
    spaced_print(
        ser.loc[stations].cal.timestamp_features(discretize=False, create_vocab=False)
    )

    ser.loc[stations].reset_index().sample(
        n=5_000, random_state=42
    ).cal.plot_across_column("started_at", "start_station_name", alpha=0.15)
    savefig("across-stations")

    print_bold("Series with calendar features")
    spaced_print(ser.loc[stations].cal.timestamp_features().groupby(level=0).head(3))

    import numpy as np
    import pymc as pm

    DAYS_IN_WEEK = 7
    HOURS_IN_DAY = 24

    day_of_week = pm.Categorical.dist(p=np.ones(DAYS_IN_WEEK) / DAYS_IN_WEEK)
    time_of_day = pm.Uniform.dist(lower=0, upper=HOURS_IN_DAY)

    N = 5_000
    kwargs = {"draws": N, "random_seed": 42}
    sample_dow, sample_tod = pm.draw([day_of_week, time_of_day], **kwargs)

    df_events = pd.DataFrame({"day_of_week": sample_dow, "hour_start": sample_tod})
    df_events["hour_end"] = df_events["hour_start"] + (5 / (24 * 60))

    from latent_calendar.plot import plot_dataframe_as_calendar
    from latent_calendar.plot.iterate import IterConfig

    config = IterConfig(start="hour_start", end="hour_end", day="day_of_week")
    ax = plot_dataframe_as_calendar(df_events, config=config, alpha=1)
    ax.set_title("Discrete day of week and continuous time of day")
    savefig("attempt-1")

    total_time_slots = DAYS_IN_WEEK * HOURS_IN_DAY
    p = np.ones(total_time_slots) / total_time_slots
    time_slot = pm.Categorical.dist(p=p)

    day_of_week = time_slot // HOURS_IN_DAY
    time_of_day = time_slot % HOURS_IN_DAY

    sample_dow, sample_tod = pm.draw([day_of_week, time_of_day], **kwargs)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    time_labeler = TimeLabeler(display=False)
    day_labeler = DayLabeler(display=False)
    for i, (station, ax) in enumerate(zip(stations, axes[0, :].ravel())):
        time_labeler = TimeLabeler(display=i == 0)
        ser.loc[station].cal.plot(
            ax=ax, day_labeler=day_labeler, time_labeler=time_labeler, alpha=0.05
        )
        ax.set_title(station)

    axes[0, 0].set_ylabel("Continuous")

    df_agg = (
        ser.loc[stations]
        .reset_index()
        .cal.aggregate_events("start_station_name", "started_at")
    )
    for i, (station, ax) in enumerate(zip(stations, axes[1, :].ravel())):
        time_labeler = TimeLabeler(display=i == 0)
        df_agg.loc[station].cal.plot_row(ax=ax, time_labeler=time_labeler)

    axes[1, 0].set_ylabel("Discrete")
    savefig("case-for-discrete")

    df_events = pd.DataFrame({"day_of_week": sample_dow, "hour_start": sample_tod})
    df_events["hour_end"] = df_events["hour_start"] + 1

    from latent_calendar.plot import plot_dataframe_as_calendar
    from latent_calendar.plot.iterate import IterConfig

    config = IterConfig(start="hour_start", end="hour_end", day="day_of_week")
    ax = plot_dataframe_as_calendar(df_events, config=config, alpha=0.15)
    ax.set_title("Discrete day of week and time of day")
    savefig("attempt-2")

    from latent_calendar.segments.hand_picked import create_box_segment, stack_segments

    mornings = create_box_segment(
        day_start=0, day_end=5, hour_start=7, hour_end=9, name="mornings"
    )
    afternoons = create_box_segment(
        day_start=0, day_end=5, hour_start=16, hour_end=19, name="afternoons"
    )
    weekends = create_box_segment(
        day_start=5, day_end=7, hour_start=10, hour_end=20, name="weekends"
    )

    df_prior = stack_segments([mornings, afternoons, weekends])
    df_prior.cal.plot_by_row()
    savefig("timeslot-prior")

    n_groups = len(df_prior)
    cluster_a = np.ones(n_groups)
    cluster_p = pm.Dirichlet.dist(a=cluster_a)
    cluster = pm.Categorical.dist(p=cluster_p, size=100)

    prior_a = df_prior.to_numpy()
    prior = pm.Dirichlet.dist(a=prior_a)

    time_slot = pm.Categorical.dist(p=prior[cluster])
    day_of_week = time_slot // HOURS_IN_DAY
    time_of_day = time_slot % HOURS_IN_DAY

    sample_dow, sample_tod = pm.draw([day_of_week, time_of_day], random_seed=42)
    df_events = pd.DataFrame({"day_of_week": sample_dow, "hour_start": sample_tod})
    df_events["hour_end"] = df_events["hour_start"] + 1

    config = IterConfig(start="hour_start", end="hour_end", day="day_of_week")
    ax = plot_dataframe_as_calendar(df_events, config=config, alpha=0.15)
    ax.set_title("Discrete day of week and time of day")
    savefig("attempt-3")

    def create_lda_sampler(
        df_behavior: pd.DataFrame,
        N: int,
        behavior_prior: float,
        timeslot_prior: float,
        timeslot_intercept: float,
    ) -> pt.TensorVariable:
        n_behaviors = len(df_behaviors)
        behavior_a = np.ones(n_behaviors) * behavior_prior
        behavior_p = pm.Dirichlet.dist(a=behavior_a)
        cluster = pm.Categorical.dist(p=behavior_p, size=N)

        prior_a = df_prior.to_numpy() * timeslot_prior + timeslot_intercept
        prior = pm.Dirichlet.dist(a=prior_a)

        return pm.Multinomial.dist(p=prior[cluster], n=1).sum(axis=0)

    df_behaviors = df_prior
    N = 5_000
    time_slot = create_lda_sampler(
        df_behavior=df_behaviors,
        N=N,
        behavior_prior=0.5,
        timeslot_prior=5,
        timeslot_intercept=0.5,
    )

    title_func = lambda idx, row: f"Sample {idx + 1}"
    pd.DataFrame(pm.draw(time_slot, draws=6, random_seed=42)).cal.plot_by_row(
        title_func=title_func
    )
    fig = plt.gcf()
    fig.suptitle(f"Random samples from the Mixture Multinomial ({N = })")
    savefig("attempt-4", fig=fig)

    partial_sampler = partial(create_lda_sampler, df_behavior=df_behaviors, N=N)
    samplers = [
        (
            "low and low",
            partial_sampler(
                behavior_prior=0.01, timeslot_prior=0.01, timeslot_intercept=0.01
            ),
        ),
        (
            "high and low",
            partial_sampler(
                behavior_prior=10, timeslot_prior=0.01, timeslot_intercept=0.01
            ),
        ),
        (
            "low and concentrated",
            partial_sampler(
                behavior_prior=0.01, timeslot_prior=10, timeslot_intercept=0.01
            ),
        ),
        (
            "low and spread",
            partial_sampler(
                behavior_prior=0.01, timeslot_prior=0.01, timeslot_intercept=10
            ),
        ),
        (
            "high and concentrated",
            partial_sampler(
                behavior_prior=10, timeslot_prior=10, timeslot_intercept=0.01
            ),
        ),
    ]
    names = [
        "Low and low",
        "High and low",
        "Low and concentrated",
        "Low and spread",
        "High and concentrated",
    ]
    n_samples = 3
    dfs = []
    for name, sampler in samplers:
        df_tmp = pd.DataFrame(pm.draw(sampler, draws=n_samples, random_seed=42))
        dfs.append(df_tmp)

    df_samples = pd.concat(dfs)
    time_labeler = TimeLabeler(stride=4)
    df_samples.cal.normalize("max").cal.plot_by_row(
        title_func=lambda idx, row: "", max_cols=n_samples, time_labeler=time_labeler
    )
    fig = plt.gcf()
    fig.suptitle(
        "Random samples from different LDA priors\nacross different behavior and timeslot priors"
    )
    for ax, name in zip(np.array(fig.axes).reshape(len(dfs), -1)[:, 0], names):
        ax.set_ylabel(name)

    savefig("different-priors", fig=fig, height=15, width=15, pad_inches=0)

    MINUTES = 60
    df_wide = df.cal.aggregate_events(
        "start_station_name", "started_at", minutes=MINUTES
    )
    df_wide = pd.concat(
        [df_wide.loc[stations], df_wide.loc[~df_wide.index.isin(stations)]]
    )

    model = LatentCalendar(n_components=3, random_state=42, n_jobs=-1)
    model.fit(df_wide.to_numpy())

    fig, ax = plt.subplots()
    ax.bar(range(model.n_components), model.component_distribution_)
    ax.set_xticks(range(model.n_components))
    ax.axhline(1 / model.n_components, color="black", linestyle="--")
    ax.set_xlabel("Latent Component")
    ax.set_ylabel("Probability")
    savefig("component-distribution", fig=fig)

    plot_model_components(model)
    savefig("model-components")

    model_8 = LatentCalendar(n_components=8, random_state=42, n_jobs=-1)
    model_8.fit(df_wide.to_numpy())

    plot_model_components(model_8, max_cols=4)
    savefig("model-components-8")

    df_wide.loc[stations].cal.plot_profile_by_row(model=model)
    fig = plt.gcf()
    for ax in np.array(fig.axes).reshape(2, 3)[:, -1]:
        ax.set_ylim(0, 1)
    savefig("station-profiles")

    df_wide.loc[stations].cal.plot_profile_by_row(model=model_8)
    savefig("station-profiles-8")

    print_bold("Different Stations")
    df_wide.loc[stations].cal.normalize("max").cal.plot_by_row()
    fig = plt.gcf()
    fig.set_figheight(6)
    fig.set_figwidth(10)
    plt.savefig("images/different-stations.png")

    axes = df_wide.loc[stations].cal.plot_profile_by_row(model=model)
    for ax in axes[:, -1].ravel():
        ax.set_ylim(0, 1)

    fig = plt.gcf()

    savefig("different-stations-profiles", fig=fig, height=12, width=20)

    df_wide = df.cal.aggregate_events(
        ["start_station_name", "after"], "started_at", minutes=MINUTES
    )

    df_stations = df_wide.loc[stations].reorder_levels([1, 0])
    df_before = df_stations.loc[False]
    df_after = df_stations.loc[True]

    df_before.cal.plot_model_predictions_by_row(df_after, model=model)
    fig = plt.gcf()
    fig.set_figheight(12)
    fig.set_figwidth(20)
    plt.savefig("images/different-stations-predictions.png")

    # Small amount of data and sensitivity to hyperparameters
    time_slot = "05 14"
    fig = plot_mock_data(time_slot, df_wide.columns, model=model)
    savefig("mock-data", fig=fig, height=5, width=20)

    time_slot = "03 07"
    fig = plot_mock_data(time_slot, df_wide.columns, model=model)
    savefig("mock-data-2", fig=fig, height=5, width=20)

    time_slot = "05 14"
    fig = plot_mock_data(time_slot, df_wide.columns, model=model_8)
    savefig("mock-data-model-2", fig=fig, height=5, width=20)

    time_slot = "01 13"
    fig = plot_mock_data(time_slot, df_wide.columns, model=model_8)
    savefig("mock-data-2-model-2", fig=fig, height=5, width=20)

    def largest_component(df: pd.DataFrame) -> pd.Series:
        largest_idx = df.iloc[-1].argmax()

        return df.iloc[:, largest_idx].rename("largest_component")

    n_events = np.array([0, 1, 2, 5, 7, 10, 20])
    df_mock = create_df_mock(n_events, df_wide.columns, time_slot)
    # Think this only has an impact on the training data
    topic_word_priors = [0.00001, 0.0001, 0.01, 0.1, 0.5, 1]
    dfs = []
    models = []
    for topic_word_prior in topic_word_priors:
        model = LatentCalendar(
            n_components=20,
            random_state=42,
            topic_word_prior=topic_word_prior,
            n_jobs=-1,
        )
        model.fit(df_wide.to_numpy())
        models.append(model)

        dfs.append(
            df_mock.cal.transform(model=model)
            .assign(largest_component=largest_component)
            .assign(topic_word_prior=topic_word_prior)
        )

    fig, axes = plt.subplots(ncols=2)

    ax = axes[0]
    plot_blank_calendar(ax=ax)
    CalendarEvent.from_vocab(time_slot).plot(ax=ax)

    df_compare = pd.concat(dfs).set_index("topic_word_prior", append=True)

    ax = axes[1]
    df_compare["largest_component"].unstack().plot(ax=ax)
    ax.set(
        ylim=(0, 1),
        xlabel="# of events",
        title="Prior regularization sensitivity",
    )
    ax.get_legend().set_title("Topic-Word prior")
    savefig("test-sensitivity-priors-topic-word")

    for model, topic_word_prior in zip(models, topic_word_priors):
        plot_model_components(model)
        fig = plt.gcf()
        fig.set_figheight(10)
        fig.set_figwidth(20)
        plt.savefig(f"images/model-components-topic-word-prior-{topic_word_prior}.png")

    # This has the impact on the predictions
    doc_topic_priors = [
        0.001,
        0.1,
        0.5,
        1,
    ]
    dfs = []
    models = []
    for doc_topic_prior in doc_topic_priors:
        model = LatentCalendar(
            n_components=3, random_state=42, doc_topic_prior=doc_topic_prior, n_jobs=-1
        )
        model.fit(df_wide.to_numpy())
        models.append(model)

        dfs.append(
            df_mock.cal.transform(model=model)
            .assign(largest_component=largest_component)
            .assign(doc_topic_prior=doc_topic_prior)
        )

    df_compare = pd.concat(dfs).set_index("doc_topic_prior", append=True)

    fig, axes = plt.subplots(ncols=2)

    ax = axes[0]
    ax.set_title("Event time slot")
    grid_lines = GridLines(dow=True, hour=True)
    plot_blank_calendar(ax=ax, grid_lines=grid_lines)
    CalendarEvent.from_vocab(time_slot).plot(
        ax=ax, fill=False, edgecolor="black", lw=2, linestyle="--"
    )

    ax = axes[1]

    df_compare["largest_component"].unstack().plot(ax=ax)
    # Make x axis labels all integers
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set(
        ylim=(0, 1),
        xlabel="# of events",
        ylabel="Probability of largest component",
        title="Prior regularization sensitivity",
    )
    ax.get_legend().set_title("Doc-topic prior")
    savefig("prior-sensitivity-doc-topic")
