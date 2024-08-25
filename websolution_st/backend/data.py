import os
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import create_engine, text

from .dictionaries import fips_to_iso

# from .exception import NoDBConnection

connection_string = os.getenv("DB_CONNECTION_STRING", "None")

"""Module for custom exceptions."""


class NoDBConnection(Exception):
    """Exception raised when there is no connection to the database."""

    pass


class DBConnector:
    def __init__(self, connection_string: str):
        """
        Initializes the DBConnector with the given connection string.

        Arguments
        ---------
        connection_string : str
            The connection string to the database.

        """
        self.connection_string = connection_string
        try:
            self.engine = create_engine(connection_string)
        except Exception:
            self.engine = None

    def fetch_data_as_dataframe(self, query: str) -> pd.DataFrame:
        """
        Returns the results of `query` as a pandas DataFrame.

        Arguments
        ---------
        query : str
            The SQL query to execute.

        Returns
        -------
        pandas.DataFrame
            The results of the query as a DataFrame.

        """
        if self.engine is None:
            raise NoDBConnection("No connection to the database.")

        with self.engine.connect() as connection:
            result = connection.execute(text(query))
            dataframe = pd.DataFrame(result.fetchall(), columns=result.keys())

        return dataframe


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


tone_prediction_gold = """
SELECT [DATE]
      ,[Country]
      ,[y_pred]
      ,[y_real]
  FROM [gkg].[tonePredictionsGold]
"""

try:
    tone_data_df = connection.fetch_data_as_dataframe(tone_prediction_gold)
    tone_data_df["DATE"] = pd.to_datetime(tone_data_df["DATE"]).dt.date
except (NoDBConnection, Exception):
    tone_data_df = None
    print("No connection to the database.")


def last_7_days() -> list[datetime]:
    """Return the last 7 days."""
    return [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]


def get_last_7_days_datasets():
    if goldstein_data_df is None or tone_data_df is None:
        return None

    last_7_days_list = last_7_days()

    goldstein_data = goldstein_data_df[
        goldstein_data_df["fecha"].isin(last_7_days_list)
    ].copy()
    tone_data = tone_data_df[tone_data_df["DATE"].isin(last_7_days_list)].copy()

    goldstein_data["iso_country"] = goldstein_data["pais"].map(fips_to_iso)
    tone_data["iso_country"] = tone_data["Country"].map(fips_to_iso)

    return goldstein_data, tone_data


def get_unique_countries():
    if goldstein_data_df is None:
        return []

    return goldstein_data_df["pais"].unique().tolist()


def get_goldstein_country(country: str):
    if goldstein_data_df is None:
        return None

    return goldstein_data_df[goldstein_data_df["pais"] == country].copy()
