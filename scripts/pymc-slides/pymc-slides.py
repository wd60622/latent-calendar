from functools import partial
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

import pandas as pd

from rich.console import Console

from lyft_bikes import read_historical_trips

from latent_calendar import datasets, LatentCalendar
from latent_calendar.plot import plot_model_components

font = {"size": 15}

matplotlib.rc("font", **font)


console = Console()


DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


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


def savefig(name: str, fig=None, figsize=(10, 20)) -> None:
    if fig is None:
        fig = plt.gcf()

    fig.set_figwidth(figsize[1])
    fig.set_figheight(figsize[0])

    plt.savefig(f"images/{name}.png")


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

    df_wide = df.cal.aggregate_events("start_station_name", "started_at", minutes=30)

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
        ["start_station_name", "after"], "started_at", minutes=30
    )

    df_stations = df_wide.loc[stations].reorder_levels([1, 0])
    df_before = df_stations.loc[False]
    df_after = df_stations.loc[True]

    df_before.cal.plot_model_predictions_by_row(df_after, model=model)
    fig = plt.gcf()
    fig.set_figheight(12)
    fig.set_figwidth(20)
    plt.savefig("images/different-stations-predictions.png")
