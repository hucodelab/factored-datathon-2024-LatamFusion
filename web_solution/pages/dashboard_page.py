import logging

import plotly.express as px
import plotly.graph_objects as go
import taipy.gui.builder as tgb
from backend import data
from taipy.gui import Markdown, State

goldstein_last_7_days, tone_last_7_days = data.get_last_7_days_datasets()
print(f"Shape of goldstein_last_7_days: {goldstein_last_7_days.shape}")
print(f"Shape of tone_last_7_days: {tone_last_7_days.shape}")

custom_color_scale = [
    [-5, "red"],  # Lowest value (red)
    [0, "yellow"],  # Intermediate color (yellow)
    [5, "green"],  # Highest value (green)
]

# Create the interactive map
fig_goldstein_map = px.choropleth(
    goldstein_last_7_days,
    locations="iso_country",
    locationmode="ISO-3",
    color="y_real",
    color_continuous_scale="blues",  # Changed to predefined colorscale
    title="World Map with y_real Index by Country",
)

fig_tone_map = px.choropleth(
    tone_last_7_days,
    locations="iso_country",
    locationmode="ISO-3",
    color="y_real",
    color_continuous_scale="blues",  # Changed to predefined colorscale
    title="World Map with y_real Index by Country",
)


average_goldstein = data.goldstein_data_df["y_real"].mean()
average_tone = data.tone_data_df["y_real"].mean()

dashboard_page = Markdown("pages/dashboard_page.md")
