import plotly.graph_objects as go
import taipy.gui.builder as tgb
from backend import data
from taipy.gui import Markdown, State

average_goldstein = data.goldstein_data_df["y_real"].mean()
average_tone = data.tone_data_df["y_real"].mean()

dashboard_page = Markdown("pages/dashboard_page.md")

