import os
import sys

import matplotlib.pyplot as plt
from library.processing import process_df_to_plot
from taipy.gui import Gui

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Process the data
train, test, real = process_df_to_plot()


# Function to plot the time series
def plot_time_series():
    fig, ax = plt.subplots()
    ax.plot(train["date"], train["goldstein"], label="Train", color="blue")
    ax.plot(test["date"], test["goldstein"], label="Test", color="orange")
    ax.plot(real["date"], real["goldstein"], label="Real", color="green")
    ax.set_xlabel("Date")
    ax.set_ylabel("Goldstein")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


# Define the Taipy GUI
page = """
# Time Series Dashboard

<|toggle_section|label=Plot|expanded=True|>

<|part|chart|>
"""

# Chart is linked to the `plot_time_series` function
chart = plot_time_series()

# Set up and run the GUI
gui = Gui(page=page, chart=chart)
gui.run()
