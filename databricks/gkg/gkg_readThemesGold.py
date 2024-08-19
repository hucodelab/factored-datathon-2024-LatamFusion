# Databricks notebook source
import numpy as np
import pandas as pd

# COMMAND ----------

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

themes = df.toPandas()
themes_list = list(themes['THEMES_EXPLODED'])

# COMMAND ----------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Ejemplo básico de clustering
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(themes_list)  # themes_list es la lista de temas

scaler = StandardScaler()
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
