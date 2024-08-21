# Databricks notebook source
# MAGIC %md ## Themes Labels: KMeans + Word2Vec
# MAGIC

# COMMAND ----------

# MAGIC %pip install nltk
# MAGIC %pip install gensim

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
from collections import Counter

# COMMAND ----------

nltk.download('punkt')
nltk.download('stopwords')

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

# Define SQL query
sql_query = "(SELECT THEMES_EXPLODED, count FROM [gkg].[THEMES] WHERE THEMES_EXPLODED != '') AS tmp"

# Load data from Azure SQL Database into a DataFrame
df = spark.read.jdbc(url=jdbc_url, table=sql_query, properties=connection_properties)

# COMMAND ----------

themes = df.toPandas()
themes_list = list(themes['THEMES_EXPLODED'])

# COMMAND ----------

codes = themes_list.copy()
tokenized_codes = [code.split('_') for code in codes]


# COMMAND ----------

stopwords_list = set(word.upper() for word in stopwords.words('english')) 
custom_stopwords = {'TAX', 'GENERAL', 'WB', 'EPU', 'USPEC', 'CRISISLEX'}

# Combine NTLK stopwords with the custom stopwords
all_stopwords = stopwords_list.union(custom_stopwords)

# Filter the lists
filtered_codes = [[word for word in code if word.upper() not in all_stopwords] for code in tokenized_codes]


# COMMAND ----------

# Word2Vec Model Training
model = Word2Vec(sentences=filtered_codes, vector_size=100, window=5, min_count=1, sg=0)


# COMMAND ----------

def get_vector_for_code(code, model):
    # Get the word vector for every word in the code
    vectors = [model.wv[word] for word in code if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# Get the vectors for all the codes
vectors = np.array([get_vector_for_code(code, model) for code in filtered_codes])

# COMMAND ----------

# Number of clusters
n_clusters = 3 

# K-Means Clusterization
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(vectors)

# Add the clusters to the codes
clustered_codes = list(zip(filtered_codes, clusters))

# COMMAND ----------

#---------------------------------
# Visualization: uncomment this section to visualice the clusters
#---------------------------------

#Reduction to 2D with t-SNE
#tsne = TSNE(n_components=2, random_state=42)
#tsne_result = tsne.fit_transform(vectors)

##df = pd.DataFrame(tsne_result, columns=['Dimension 1', 'Dimension 2'])
#df['Cluster'] = clusters

#plt.figure(figsize=(10, 8))
#sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='Cluster', palette='viridis', data=df, s=100, alpha=0.7)
#plt.title('Visualización de Clusters usando t-SNE')
#plt.legend(title='Cluster')
#plt.show()

#---------------------------------
# End of Visualization: uncomment this section to visualice the clusters
#---------------------------------

# COMMAND ----------

# DataFrame containing the codes with their respective clusters
df = pd.DataFrame({'Code': filtered_codes, 'Cluster': clusters})

cluster_word_counts = {}

for cluster_id in range(n_clusters):
    cluster_codes = df[df['Cluster'] == cluster_id]['Code']
    all_words = [word for code in cluster_codes for word in code]
    word_counts = Counter(all_words)
    cluster_word_counts[cluster_id] = word_counts



# COMMAND ----------

# Dictionary mapping the clusters clasification:
cluster_names = {
    0: 'ECONOMIC',
    1: 'SOCIAL',
    2: 'POLITICAL'
}

df['Cluster_Name'] = df['Cluster'].map(cluster_names)

df.head()

# COMMAND ----------

len(df)

# COMMAND ----------


df_clusters = pd.DataFrame({'THEMES_EXPLODED': themes_list, 'Cluster': clusters})

df_clusters['Cluster_Name'] = df_clusters['Cluster'].map(cluster_names)

themes_mapping = df_clusters.set_index('THEMES_EXPLODED')['Cluster_Name'].to_dict()

themes['Cluster_Name'] = themes['THEMES_EXPLODED'].map(themes_mapping)

print(themes.head())



# COMMAND ----------

df_clusters = pd.DataFrame({'THEMES_EXPLODED': themes_list, 'Cluster_Name': df['Cluster_Name']})

# Convertir el DataFrame en un diccionario
themes_mapping = df_clusters.set_index('THEMES_EXPLODED')['Cluster_Name'].to_dict()

# Imprimir el diccionario
print(themes_mapping)

# COMMAND ----------

# MAGIC %md ## Themes Labels: Code Modularization:
# MAGIC

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
from collections import Counter

# COMMAND ----------

def install_dependencies():
    %pip install nltk
    %pip install gensim
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')

def configure_azure_storage(storage_account_name, storage_account_key):
    spark.conf.set(
        f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net",
        f"{storage_account_key}"
    )

def read_csv_from_blob(storage_account_name, container_name, file_name):
    file_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/{file_name}"
    return spark.read.format("csv").option("header", "true").load(file_path).dropna(subset=["THEMES_EXPLODED"])

def configure_azure_sql(jdbc_hostname, jdbc_port, jdbc_database, user, password):
    jdbc_url = f"jdbc:sqlserver://{jdbc_hostname}:{jdbc_port};database={jdbc_database}"
    connection_properties = {
        "user": user,
        "password": password,
        "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
    }
    return jdbc_url, connection_properties

def read_data_from_sql(jdbc_url, sql_query, connection_properties):
    return spark.read.jdbc(url=jdbc_url, table=sql_query, properties=connection_properties).toPandas()

def preprocess_themes(themes_list):
    tokenized_codes = [code.split('_') for code in themes_list]
    stopwords_list = set(word.upper() for word in stopwords.words('english')) 
    custom_stopwords = {'TAX', 'GENERAL', 'WB', 'EPU', 'USPEC', 'CRISISLEX'}
    all_stopwords = stopwords_list.union(custom_stopwords)
    filtered_codes = [[word for word in code if word.upper() not in all_stopwords] for code in tokenized_codes]
    return filtered_codes
  

def train_word2vec_model(filtered_codes, vector_size=100, window=5, min_count=1, sg=0):
    return Word2Vec(sentences=filtered_codes, vector_size=vector_size, window=window, min_count=min_count, sg=sg)

def get_vector_for_code(code, model):
    vectors = [model.wv[word] for word in code if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)
  
def cluster_codes(filtered_codes, model, n_clusters=3):
    vectors = np.array([get_vector_for_code(code, model) for code in filtered_codes])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(vectors)
    return clusters

def classify_clusters(filtered_codes, clusters, n_clusters=3):
    df = pd.DataFrame({'Code': filtered_codes, 'Cluster': clusters})
    cluster_word_counts = {}
    for cluster_id in range(n_clusters):
        cluster_codes = df[df['Cluster'] == cluster_id]['Code']
        all_words = [word for code in cluster_codes for word in code]
        word_counts = Counter(all_words)
        cluster_word_counts[cluster_id] = word_counts
    return cluster_word_counts

def assign_cluster_names(clusters):
    cluster_names = {
        0: 'ECONOMIC',
        1: 'SOCIAL',
        2: 'POLITICAL'
    }
    return pd.DataFrame({'Cluster': clusters}).replace({"Cluster": cluster_names})

def visualize_clusters(vectors, clusters):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(vectors)
    df = pd.DataFrame(tsne_result, columns=['Dimension 1', 'Dimension 2'])
    df['Cluster'] = clusters

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='Cluster', palette='viridis', data=df, s=100, alpha=0.7)
    plt.title('Visualización de Clusters usando t-SNE')
    plt.legend(title='Cluster')
    plt.show()

def get_themes_mapping(themes_list, cluster_names):
    # Crear un DataFrame con las columnas 'THEMES_EXPLODED' y 'Cluster_Name'
    df_clusters = pd.DataFrame({'THEMES_EXPLODED': themes_list, 'Cluster_Name': cluster_names})
    
    # Convertir el DataFrame en un diccionario
    themes_mapping = df_clusters.set_index('THEMES_EXPLODED')['Cluster_Name'].to_dict()
    
    return themes_mapping



def main():
    install_dependencies()

    storage_account_key = "wgbe0Fzs4W3dPNc35dp//uumz+SPDXVLLGu0mNaxTs2VLHCCPnD7u79PYt4mKeSFboqMRnZ+s+ez+ASty+k+sQ==" #QUITAR ESTO
    storage_account_name = "factoredatathon2024"
    container_name = "gold"
    configure_azure_storage(storage_account_name, storage_account_key)

    # Read CSV from Azure Blob Storage
    df_blob = read_csv_from_blob(storage_account_name, container_name, "gkg/themesSortedGold.csv")

    # Read the data from Azure SQL
    jdbc_hostname = "factoredata2024.database.windows.net"
    jdbc_port = 1433
    jdbc_database = "dactoredata2024"
    user = "factoredata2024admin"
    password = "mdjdmliipo3^%^$5mkkm63" #QUITAR ESTO
    jdbc_url, connection_properties = configure_azure_sql(jdbc_hostname, jdbc_port, jdbc_database, user, password)
    sql_query = "(SELECT THEMES_EXPLODED, count FROM [gkg].[THEMES] WHERE THEMES_EXPLODED != '') AS tmp"
    themes = read_data_from_sql(jdbc_url, sql_query, connection_properties)
    
    themes_list = list(themes['THEMES_EXPLODED'])
    filtered_codes = preprocess_themes(themes_list)
    
    model = train_word2vec_model(filtered_codes)
    
    clusters = cluster_codes(filtered_codes, model)
    cluster_word_counts = classify_clusters(filtered_codes, clusters)
    
    cluster_df = assign_cluster_names(clusters)
    themes['Cluster_Name'] = cluster_df['Cluster']
    
    print(themes.head())

    themes_mapping = get_themes_mapping(themes_list, df['Cluster_Name'])

    print(themes_mapping)



# COMMAND ----------

spark_themes = spark.createDataFrame(themes)

# COMMAND ----------

dbfs_path_delta = "/mnt/silver/themesMappedSilver"
spark_themes.write.format("delta").mode("overwrite").save(dbfs_path_delta)

# COMMAND ----------

main()
