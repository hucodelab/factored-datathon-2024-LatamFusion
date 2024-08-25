import os

from .db import DBConnector
from .exception import NoDBConnection

connection_string = os.getenv("DB_CONNECTION_STRING", "None")

connection = DBConnector(connection_string)

goldstein_prediction_gold = """
SELECT fecha, pais, y_pred, y_real
FROM dactoredata2024.events.goldsteinPredictionsGold
"""

try:
    goldstein_data_df = connection.fetch_data_as_dataframe(goldstein_prediction_gold)

except (NoDBConnection, Exception):
    goldstein_data_df = None
    print("No connection to the database.")


def get_goldstein_country(country: str):
    if goldstein_data_df is None:
        return None

    return goldstein_data_df[goldstein_data_df["pais"] == country].copy()
