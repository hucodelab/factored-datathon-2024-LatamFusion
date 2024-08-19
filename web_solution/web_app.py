import plotly.graph_objects as go
import taipy.gui.builder as tgb
from library.processing import process_df_to_plot
from taipy.gui import Gui

datasets_dict = process_df_to_plot()

# ts_plot_config = {
#     "name_df": datasets_dict.keys(),
#     "names": datasets_dict.keys().mapping(lambda x: x.capitalize()),
#     "colors": ["blue", "orange", "green"],
# }


def create_time_series_plot():
    # Extract datasets
    train_df = datasets_dict["train"]
    test_df = datasets_dict["test"]
    real_df = datasets_dict["real"]

    trace_train = go.Scatter(
        x=train_df["date"],
        y=train_df["goldstein"],
        mode="lines",
        name="Train",
        line=dict(color="blue"),
    )

    trace_test = go.Scatter(
        x=test_df["date"],
        y=test_df["goldstein"],
        mode="lines",
        name="Test",
        line=dict(color="orange"),
    )

    trace_real = go.Scatter(
        x=real_df["date"],
        y=real_df["goldstein"],
        mode="lines",
        name="Real",
        line=dict(color="green"),
    )

    fig = go.Figure(data=[trace_train, trace_test, trace_real])

    return fig


fig = create_time_series_plot()

with tgb.Page() as page:
    tgb.text("# Latam Fusion Dashboard", mode="md")
    tgb.chart(figure="{fig}")

Gui(page).run(host="0.0.0.0", port=5000)
