import pandas as pd 

import matplotlib.pyplot as plt

import latent_calendar


if __name__ == "__main__": 
    file = "https://raw.githubusercontent.com/posit-dev/great-tables/main/great_tables/data/05-pizzaplace.csv"

    df = pd.read_csv(file)

    # Create a datetime column
    df["datetime"] = pd.to_datetime(df["date"].str.cat(df["time"], sep=" "))

    # Take advantage of the cal attribute for plotting
    df_plot = (
        df
        .assign(title="Pizza ordering times")
        .cal.aggregate_events(
            by="title", 
            timestamp_col="datetime", 
            minutes=30
        )
    )
    ax = (
        df_plot
        .cal.plot_by_row()
    )
    plt.show()

