from functools import partial
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from rich.console import Console

from lyft_bikes import read_historical_trips

from latent_calendar import datasets, LatentCalendar
from latent_calendar.plot import plot_model_components, TimeLabeler, CalendarEvent
from latent_calendar.vocab import make_human_readable

font = {"size": 15}

matplotlib.rc("font", **font)


console = Console()


HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR = HERE / "images"
IMAGE_DIR.mkdir(exist_ok=True)

START_DATE = "2023-05-01"
END_DATE = "2023-07-31"
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


def savefig(name: str, fig=None, height: float = 12, width: float = 20) -> None:
    if fig is None:
        fig = plt.gcf()

    fig.set_figwidth(width)
    fig.set_figheight(height)

    file = IMAGE_DIR / f"{name}.png"
    plt.savefig(file)


def main() -> None:
    pass


if __name__ == "__main__":
    df = read_chicago_trips()

    df["weekofyear"] = df["started_at"].dt.isocalendar().week
    df["after"] = df["weekofyear"].ge(27)

    ser = (
        df.dropna(subset=["start_station_name"])
        .set_index(["start_station_name"], append=True)
        .loc[:, "started_at"]
        .reorder_levels([1, 0])
        .sort_index()
    )

    print_bold("Series")
    spaced_print(ser)

    print_bold("Series with calendar features")
    spaced_print(ser.cal.timestamp_features())

    MINUTES = 60
    df_wide = df.cal.aggregate_events("start_station_name", "started_at", minutes=MINUTES)

    model = LatentCalendar(n_components=3, random_state=42, n_jobs=-1)
    model.fit(df_wide.to_numpy())

    plot_model_components(model)
    fig = plt.gcf()
    fig.set_figheight(10)
    fig.set_figwidth(20)
    plt.savefig("images/model-components.png")

    print_bold("Different Stations")
    stations = ["DuSable Lake Shore Dr & North Blvd", "Clark St & Schiller St"]
    df_wide.loc[stations].cal.normalize("max").cal.plot_by_row()
    fig = plt.gcf()
    fig.set_figheight(6)
    fig.set_figwidth(10)
    plt.savefig("images/different-stations.png")

    axes = df_wide.loc[stations].cal.plot_profile_by_row(model=model)
    for ax in axes[:, -1].ravel():
        ax.set_ylim(0, 1)

    fig = plt.gcf()
    fig.set_figheight(12)
    fig.set_figwidth(20)
    plt.savefig("images/different-stations-profiles.png")

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
    ncols = df_wide.shape[1]
    
    n_events = np.array([0, 1, 2, 5, 7, 10, 20])
    df_mock = pd.DataFrame(np.zeros((len(n_events), ncols)), columns=df_wide.columns, index=n_events)

    time_slot = "05 14"
    df_mock.loc[:, time_slot] = n_events

    fig, axes = plt.subplots(ncols=4)
    fig.suptitle(f"Events at {make_human_readable(time_slot)}")

    df_mock_predictions = df_mock.cal.predict(model=model)
    event = CalendarEvent.from_vocab(time_slot)
    for idx, (ax, n_events_idx) in enumerate(zip(axes, [0, 1, -1])): 
        ax.set(
            title=f"{n_events[n_events_idx]} events",
        )
        time_labeler = TimeLabeler(display= idx == 0)
        df_mock_predictions.iloc[n_events_idx].cal.plot_row(ax=ax, time_labeler=time_labeler)
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
    savefig("mock-data", fig=fig, height=8, width=20)

    def largest_component(df: pd.DataFrame) -> pd.Series: 
        largest_idx = df.iloc[-1].argmax()

        return df.iloc[:, largest_idx].rename("largest_component")


    # Think this only has an impact on the training data
    topic_word_priors = [
        0.00001, 0.0001, 0.01, 0.1, 0.5, 1 
    ]
    dfs = []
    models = []
    for topic_word_prior in topic_word_priors:
        model = LatentCalendar(n_components=3, random_state=42, topic_word_prior=topic_word_prior)
        model.fit(df_wide.to_numpy())
        models.append(model)

        dfs.append(
            df_mock
            .cal.transform(model=model)
            .assign(largest_component=largest_component)
            .assign(topic_word_prior=topic_word_prior)
        )
    
    df_compare = pd.concat(dfs).set_index("topic_word_prior", append=True)
    ax = df_compare["largest_component"].unstack().plot()
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
        0.001, 0.1, 0.5, 1,
    ]
    dfs = []
    for doc_topic_prior in doc_topic_priors:
        model = LatentCalendar(n_components=3, random_state=42, doc_topic_prior=doc_topic_prior)
        model.fit(df_wide.to_numpy())

        dfs.append(
            df_mock
            .cal.transform(model=model)
            .assign(largest_component=largest_component)
            .assign(doc_topic_prior=doc_topic_prior)
        )
    
    df_compare = pd.concat(dfs).set_index("doc_topic_prior", append=True)

    ax = df_compare["largest_component"].unstack().plot()
    ax.set(
        ylim=(0, 1), 
        xlabel="# of events", 
        ylabel="Probability of largest component",
        title="Prior regularization sensitivity",
    )
    ax.get_legend().set_title("Doc-topic prior")
    savefig("test-sensitivity-priors")