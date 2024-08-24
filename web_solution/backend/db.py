import pandas as pd
from sqlalchemy import create_engine, text


class DBConnector:
    def __init__(self, connection_string):
        """
        Initializes the DBConnector with the given connection string.

        Arguments
        ---------
        connection_string : str
            The connection string to the database.

        """
        self.connection_string = connection_string
        self.engine = create_engine(connection_string)

    def fetch_data_as_dataframe(self, query):
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

        # Execute query and fetch data into a pandas DataFrame
        with self.engine.connect() as connection:
            result = connection.execute(text(query))
            dataframe = pd.DataFrame(result.fetchall(), columns=result.keys())

        return dataframe
