import plotly.graph_objects as go
import taipy.gui.builder as tgb
from backend import data
from taipy.gui import State


def create_time_series_plot(country_code: str):
    data_for_plot = data.get_goldstein_country(country_code)

    if data_for_plot is None:
        return go.Figure()
    else:
        trace_pred = go.Scatter(
            x=data_for_plot["fecha"],
            y=data_for_plot["y_pred"],
            mode="lines",
            name="Predicted",
            line=dict(color="blue"),
        )

        trace_real = go.Scatter(
            x=data_for_plot["fecha"],
            y=data_for_plot["y_real"],
            mode="lines",
            name="Real",
            line=dict(color="orange"),
        )

        fig = go.Figure(data=[trace_pred, trace_real])

        return fig


goldstein_data_shape = (
    "No value generated"
    if data.goldstein_data_df is None
    else data.goldstein_data_df.shape
)

# Initialize the figure for the default country
selected_country = "NL"  # Default country
fig = create_time_series_plot(selected_country)

countries = data.get_unique_countries()


def on_country_change(state: State, country_code: str):
    state.selected_country = country_code
    state.fig = create_time_series_plot(country_code)


with tgb.Page() as dashboard_page:
    tgb.text("# Latam Fusion Dashboard", mode="md")
    tgb.text(f"Goldstein dataset shape: {goldstein_data_shape}")
    # Add the dropdown to select the country
    tgb.selector(
        id="country_dropdown",
        label="Select Country",
        value=selected_country,
        lov=[{"label": country, "value": country} for country in countries],
        on_change=on_country_change,
        dropdown=True,
    )
    tgb.chart(figure="{fig}")
