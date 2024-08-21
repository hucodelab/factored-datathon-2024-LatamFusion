# Databricks notebook source
import numpy as np
import pandas as pd

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
#sql_query = "(SELECT THEMES_EXPLODED, count FROM [gkg].[THEMES] WHERE THEMES_EXPLODED != '' AND count > 100) AS tmp"
sql_query = "(SELECT THEMES_EXPLODED, count FROM [gkg].[THEMES] WHERE THEMES_EXPLODED != '') AS tmp"

# Load data from Azure SQL Database into a DataFrame
df = spark.read.jdbc(url=jdbc_url, table=sql_query, properties=connection_properties)

# COMMAND ----------

df.show()

# COMMAND ----------

themes = df.toPandas()


# COMMAND ----------

themes_list = list(themes['THEMES_EXPLODED'])
themes_list

# COMMAND ----------

themes.head()

# COMMAND ----------

# MAGIC %md ## TfidfVectorizer + Kmeans + DBSCAN
# MAGIC

# COMMAND ----------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(themes_list)  # themes_list es la lista de temas

kmeans = KMeans(n_clusters=10, init='k-means++', max_iter=500, n_init=10, random_state=42)
kmeans.fit(X)

themes['Cluster'] = kmeans.labels_

# COMMAND ----------

themes['Cluster'].value_counts()

# COMMAND ----------

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# Aplicar t-SNE para reducir la dimensionalidad a 2D
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X.toarray())

# Convertir el resultado de t-SNE en un DataFrame para una manipulación más sencilla
tsne_df = pd.DataFrame(X_tsne, columns=['Dim1', 'Dim2'])
tsne_df['Cluster'] = themes['Cluster']

# Visualizar los clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_df['Dim1'], tsne_df['Dim2'], c=tsne_df['Cluster'], cmap='viridis', s=10, alpha=0.6)

# Añadir leyenda con los clusters
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.title('Clusters visualizados con t-SNE')
plt.xlabel('Dim1')
plt.ylabel('Dim2')
plt.show()

# COMMAND ----------

from sklearn.cluster import DBSCAN

sample_X = X # Tomar una muestra de los primeros 10,000 temas
dbscan = DBSCAN(eps=0.1, min_samples=20)
labels = dbscan.fit_predict(sample_X)


# COMMAND ----------

labels_df = pd.DataFrame(labels, columns=['Cluster'])

# Contar los valores de cada cluster
label_counts = labels_df['Cluster'].value_counts()

# Mostrar los resultados
label_counts

# COMMAND ----------

themes.head()

# COMMAND ----------

themes['Cluster'].value_counts()


# COMMAND ----------

# Aplicar t-SNE para reducir la dimensionalidad a 2D
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(sample_X.toarray())

# Convertir el resultado de t-SNE en un DataFrame para una manipulación más sencilla
tsne_df = pd.DataFrame(X_tsne, columns=['Dim1', 'Dim2'])
tsne_df['Cluster'] = labels_df['Cluster']

# Visualizar los clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_df['Dim1'], tsne_df['Dim2'], c=tsne_df['Cluster'], cmap='viridis', s=10, alpha=0.6)

# Añadir leyenda con los clusters
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.title('Clusters visualizados con t-SNE')
plt.xlabel('Dim1')
plt.ylabel('Dim2')
plt.show()

# COMMAND ----------

len(themes.query('Cluster == 0'))

# COMMAND ----------

# df.count()
# 54912 themes

# COMMAND ----------

# MAGIC %md ## Word2Vec Approach
# MAGIC

# COMMAND ----------

# MAGIC %pip install nltk

# COMMAND ----------

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# COMMAND ----------

nltk.download('punkt')
nltk.download('stopwords')

# COMMAND ----------

codes = themes_list.copy()

# Descomposición por el separador "_"
tokenized_codes = [code.split('_') for code in codes]

print(tokenized_codes)

# COMMAND ----------

from nltk.corpus import stopwords
# Lista de stopwords y palabras a eliminar

# Descargar las stopwords de NLTK si no las tienes ya
nltk.download('stopwords')

# Obtener la lista de stopwords en inglés y agregar 'TAX'
stopwords_list = set(word.upper() for word in stopwords.words('english')) 
custom_stopwords = {'TAX', 'GENERAL', 'WB', 'EPU', 'USPEC', 'CRISISLEX'}

# Combina las stopwords de NLTK con las custom stopwords
all_stopwords = stopwords_list.union(custom_stopwords)

# Filtrar las listas eliminando las stopwords y la palabra 'TAX'
filtered_codes = [[word for word in code if word.upper() not in all_stopwords] for code in tokenized_codes]

print(filtered_codes)


# COMMAND ----------

# MAGIC %pip install gensim

# COMMAND ----------

from gensim.models import Word2Vec

# Entrenamiento del modelo Word2Vec
model = Word2Vec(sentences=filtered_codes, vector_size=100, window=5, min_count=1, sg=0)


# COMMAND ----------

def get_vector_for_code(code, model):
    # Obtener vectores de palabras para cada palabra en el código
    vectors = [model.wv[word] for word in code if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# Obtener vectores para todos los códigos
vectors = np.array([get_vector_for_code(code, model) for code in filtered_codes])

# COMMAND ----------

# Número de clusters
n_clusters = 3  # Ajusta este valor según sea necesario

# Aplicar K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(vectors)

# Agregar los clusters a los códigos
clustered_codes = list(zip(filtered_codes, clusters))

# COMMAND ----------

import seaborn as sns
#Reducción a 2D con t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(vectors)

df = pd.DataFrame(tsne_result, columns=['Dimension 1', 'Dimension 2'])
df['Cluster'] = clusters

# Visualización
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='Cluster', palette='viridis', data=df, s=100, alpha=0.7)
plt.title('Visualización de Clusters usando t-SNE')
plt.legend(title='Cluster')
plt.show()


# COMMAND ----------

from collections import Counter
import pandas as pd

# Crear un DataFrame con los códigos y sus respectivos clusters
df = pd.DataFrame({'Code': filtered_codes, 'Cluster': clusters})

# Crear un diccionario para contar las palabras en cada cluster
cluster_word_counts = {}

for cluster_id in range(n_clusters):
    # Obtener todos los temas en el cluster
    cluster_codes = df[df['Cluster'] == cluster_id]['Code']
    
    # Aplanar la lista de listas de temas en una sola lista de palabras
    all_words = [word for code in cluster_codes for word in code]
    
    # Contar las palabras
    word_counts = Counter(all_words)
    
    # Guardar el conteo en el diccionario
    cluster_word_counts[cluster_id] = word_counts

# Mostrar las palabras más comunes en cada cluster
for cluster_id, word_counts in cluster_word_counts.items():
    print(f"Cluster {cluster_id}:")
    for word, count in word_counts.most_common(10):  # Mostrar las 10 palabras más comunes
        print(f"  {word}: {count}")
    print()

# COMMAND ----------


# Crear un DataFrame con los códigos y sus respectivos clusters
df = pd.DataFrame({'Code': filtered_codes, 'Cluster': clusters})

# Crear un diccionario que mapea los clusters a los nombres deseados
cluster_names = {
    0: 'ECONOMIC',   # Ajusta los números según tus clusters
    1: 'SOCIAL',
    2: 'POLITICAL'
}

# Reemplazar los identificadores de cluster por los nombres
df['Cluster_Name'] = df['Cluster'].map(cluster_names)

# Mostrar el DataFrame con los nombres de los clusters
print(df.head())

# Opcional: Contar las palabras en cada nuevo cluster con nombre
cluster_word_counts = {}
for cluster_name in cluster_names.values():
    # Obtener todos los temas en el cluster
    cluster_codes = df[df['Cluster_Name'] == cluster_name]['Code']
    
    # Aplanar la lista de listas de temas en una sola lista de palabras
    all_words = [word for code in cluster_codes for word in code]
    
    # Contar las palabras
    word_counts = Counter(all_words)
    
    # Guardar el conteo en el diccionario
    cluster_word_counts[cluster_name] = word_counts

# Mostrar las palabras más comunes en cada nuevo cluster
for cluster_name, word_counts in cluster_word_counts.items():
    print(f"Cluster '{cluster_name}':")
    for word, count in word_counts.most_common(10):  # Mostrar las 10 palabras más comunes
        print(f"  {word}: {count}")
    print()

# COMMAND ----------

import pandas as pd

# Crear un DataFrame con los temas y sus respectivos clusters
df_clusters = pd.DataFrame({'THEMES_EXPLODED': themes_list, 'Cluster': clusters})

# Crear un diccionario para mapear los códigos a los nombres de clusters
cluster_names = {
    0: 'ECONOMIC',   # Ajusta los números según tus clusters
    1: 'SOCIAL',
    2: 'POLITICAL'
}

# Mapea el identificador del cluster al nombre
df_clusters['Cluster_Name'] = df_clusters['Cluster'].map(cluster_names)

# Crear un diccionario de mapeo de THEMES_EXPLODED a Cluster_Name
themes_mapping = df_clusters.set_index('THEMES_EXPLODED')['Cluster_Name'].to_dict()

# Agregar la columna Cluster_Name al DataFrame original 'themes'
themes['Cluster_Name'] = themes['THEMES_EXPLODED'].map(themes_mapping)

# Mostrar el DataFrame con la nueva columna
print(themes.head())



# COMMAND ----------

# MAGIC %md ##Intento de ayer
# MAGIC

# COMMAND ----------

# Función para obtener el vector promedio de un código
def get_code_vector(code):
    parts = code.split('_')
    vectors = [model.wv[part] for part in parts if part in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)  # Vector cero si no hay tokens en el vocabulario

# Ejemplo de uso
vector_soc_generalcrime = get_code_vector('SOC_GENERALCRIME')
vector_arrest = get_code_vector('ARREST')

# COMMAND ----------

df = pd.DataFrame(tsne_result, columns=['Dimension 1', 'Dimension 2'])
df['Cluster'] = clusters

# Visualización
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='Cluster', palette='viridis', data=df, s=100, alpha=0.7)
plt.title('Visualización de Clusters usando t-SNE')
plt.legend(title='Cluster')
plt.show()

# COMMAND ----------

from sklearn.cluster import KMeans

# Supón que `all_vectors` es una lista de todos los vectores de tus códigos
all_vectors = [
    get_code_vector(code) for code in codes
]

# Número de clústeres (ajusta según tus necesidades)
num_clusters = 5

# Entrenamiento del modelo K-means
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_vectors)

# Asignación de etiquetas a cada código
labels = kmeans.labels_

print(labels)

# COMMAND ----------

labels_df = pd.DataFrame(labels, columns=['Cluster'])

# Contar los valores de cada cluster
label_counts = labels_df['Cluster'].value_counts()

# Mostrar los resultados
label_counts

# COMMAND ----------

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Supón que `all_vectors` es una lista de todos los vectores de tus códigos
all_vectors = [
    get_code_vector(code) for code in codes
]

# Aplicar t-SNE para reducir la dimensionalidad
tsne = TSNE(n_components=2, random_state=0)
reduced_vectors = tsne.fit_transform(all_vectors)

# COMMAND ----------

# Supón que `labels` es una lista de etiquetas de cluster para cada código
labels = kmeans.labels_

# Crear el gráfico
plt.figure(figsize=(10, 7))
scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels, cmap='viridis', marker='o')

# Agregar una barra de color para mostrar las etiquetas de cluster
plt.colorbar(scatter)

# Agregar etiquetas y título
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.title('Visualización de Clusters con t-SNE')

# Mostrar el gráfico
plt.show()

# COMMAND ----------

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Aplicar PCA para reducir a 3D
pca = PCA(n_components=3)
reduced_vectors_3d = pca.fit_transform(all_vectors)

# Crear el gráfico 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(reduced_vectors_3d[:, 0], reduced_vectors_3d[:, 1], reduced_vectors_3d[:, 2], c=labels, cmap='viridis', marker='o')

# Agregar una barra de color para mostrar las etiquetas de cluster
plt.colorbar(scatter)

# Agregar etiquetas y título
ax.set_xlabel('Componente 1')
ax.set_ylabel('Componente 2')
ax.set_zlabel('Componente 3')
ax.set_title('Visualización de Clusters en 3D')

# Mostrar el gráfico
plt.show()

# COMMAND ----------

from collections import Counter
import numpy as np

# Supón que `tokenized_codes` es la lista de listas de tokens obtenida anteriormente
# y `labels` es la lista de etiquetas de clúster para cada código

# Crear un diccionario para almacenar las palabras por clúster
cluster_words = {i: [] for i in set(labels)}

# Rellenar el diccionario con las palabras de cada clúster
for code, label in zip(tokenized_codes, labels):
    cluster_words[label].extend(code)

# Contar la frecuencia de cada palabra en cada clúster
cluster_word_counts = {cluster: Counter(words) for cluster, words in cluster_words.items()}

# Encontrar las 5 palabras más frecuentes en cada clúster
top_words = {cluster: counter.most_common(5) for cluster, counter in cluster_word_counts.items()}

# Imprimir los resultados
for cluster, words in top_words.items():
    print(f'Cluster {cluster}:')
    for word, freq in words:
        print(f'  "{word}" con frecuencia {freq}')
    print()

