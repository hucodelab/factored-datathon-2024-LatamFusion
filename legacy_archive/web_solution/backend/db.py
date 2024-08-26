import pandas as pd
from sqlalchemy import create_engine, text

from .exception import NoDBConnection


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
