# Databricks notebook source
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# COMMAND ----------

# HERE IS READING FROM READING FROM STORAGE ACCOUNT
storage_account_key = "wgbe0Fzs4W3dPNc35dp//uumz+SPDXVLLGu0mNaxTs2VLHCCPnD7u79PYt4mKeSFboqMRnZ+s+ez+ASty+k+sQ=="
storage_account_name = "factoredatathon2024"
container_name = "gold"

spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net",
    f"{storage_account_key}"
)

file_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/gkg/themesSortedGold.csv"
df = spark.read.format("csv").option("header", "true").load(file_path)
df = df.dropna(subset=["THEMES_EXPLODED"])

# COMMAND ----------

### HERE YOU CAN READ FROM AZURE SQL
# Define the JDBC URL
jdbc_hostname = "factoredata2024.database.windows.net"
jdbc_port = 1433
jdbc_database = "dactoredata2024"
jdbc_url = f"jdbc:sqlserver://{jdbc_hostname}:{jdbc_port};database={jdbc_database}"

# Define the connection properties
connection_properties = {
    "user": "factoredata2024admin",
    "password": "mdjdmliipo3^%^$5mkkm63",
    "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

# Define your SQL query
sql_query = "(SELECT THEMES_EXPLODED FROM [gkg].[THEMES] WHERE THEMES_EXPLODED != '') AS tmp"

# Load data from Azure SQL Database into a DataFrame
df = spark.read.jdbc(url=jdbc_url, table=sql_query, properties=connection_properties)

# COMMAND ----------

df.show()

# COMMAND ----------

themes = df.toPandas()
themes_list = list(themes['THEMES_EXPLODED'])

# COMMAND ----------

themes_list

# COMMAND ----------


# Ejemplo básico de clustering
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(themes_list)  # themes_list es la lista de temas

#scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)  # X es tu matriz TF-IDF o embeddings

# kmeans = KMeans(n_clusters=10, random_state=42)
# kmeans.fit(X_scaled)

kmeans = KMeans(n_clusters=10, init='k-means++', max_iter=500, n_init=10, random_state=42)
kmeans.fit(X)

# kmeans = KMeans(n_clusters=10, random_state=42)  # Ajusta el número de clusters según necesites
# kmeans.fit(X)

# Agrega el cluster resultante al DataFrame
themes['Cluster'] = kmeans.labels_

# COMMAND ----------

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans.labels_, cmap='viridis')
plt.show()

# COMMAND ----------

len(themes.query('Cluster == 0'))

# COMMAND ----------

# df.count()
# 54912 themes

# COMMAND ----------

X

# COMMAND ----------

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# COMMAND ----------

themes = pd.DataFrame(themes_list )
themes.columns = ["themes"]

# COMMAND ----------

themes

# COMMAND ----------

themes_list

# COMMAND ----------

themes['THEMES_EXPLODED'] = themes['THEMES_EXPLODED'].apply(lambda x: x.split(','))

# COMMAND ----------

themes_list = themes['THEMES_EXPLODED'].to_list()

# COMMAND ----------



# COMMAND ----------


