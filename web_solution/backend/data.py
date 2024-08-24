import os

from .db import DBConnector

connection_string = os.environ["DB_CONNECTION_STRING"]

connection = DBConnector(connection_string)

goldstein_prediction_gold = """
SELECT fecha, pais, y_pred, y_real
FROM dactoredata2024.events.goldsteinPredictionsGold
"""

goldstein_data_df = connection.fetch_data_as_dataframe(goldstein_prediction_gold)
