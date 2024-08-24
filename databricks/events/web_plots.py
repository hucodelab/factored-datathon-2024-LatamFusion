# Databricks notebook source
 jdbc_hostname = "factoredata2024.database.windows.net"
jdbc_port = 1433
jdbc_database = "dactoredata2024"
jdbc_url = f"jdbc:sqlserver://{jdbc_hostname}:{jdbc_port};database={jdbc_database}"

# Define the connection properties
connection_properties = {
    "user": "factoredata2024admin",
    "password": dbutils.secrets.get(scope="events", key="ASQLPassword"),
    "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

