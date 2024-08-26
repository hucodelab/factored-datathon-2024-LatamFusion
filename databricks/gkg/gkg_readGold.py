# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import timedelta

spark = SparkSession.builder.appName("AzureDataAnalysis").getOrCreate()
storage_account_name = "factoredatathon2024"
storage_account_key = "wgbe0Fzs4W3dPNc35dp//uumz+SPDXVLLGu0mNaxTs2VLHCCPnD7u79PYt4mKeSFboqMRnZ+s+ez+ASty+k+sQ=="
container_name = "silver"
output_file = "gkg_model_predictions.csv"

storage_account_key = dbutils.secrets.get(scope="events", key="DataLakeKey")
spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net",
    f"{storage_account_key}"
)

file_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/{file_name}"


jdbc_hostname = "factoredata2024.database.windows.net"
jdbc_port = 1433
jdbc_database = "dactoredata2024"
jdbc_url = f"jdbc:sqlserver://{jdbc_hostname}:{jdbc_port};database={jdbc_database}"

connection_properties = {
    "user": "factoredata2024admin",
    "password": "mdjdmliipo3^%^$5mkkm63",
    "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}


table_name = "gkg.tonePredictionsGold"
df = spark.read.format("csv").option("header", "True").load(file_path)
column_names = ["DATE", "Country", "y_pred", "y_real"]
events = df
events = events.toDF(*column_names).toPandas()


------------------
# Convertir DATE a datetime
events['DATE'] = pd.to_datetime(events['DATE'])

# DataFrame para almacenar todas las predicciones
df_combined = pd.DataFrame()

# Obtener la lista de países únicos
all_countries = events['Country'].unique()
#countries = all_countries[:150]
countries = all_countries[:]
# Iterar sobre cada país
for country_selected in countries:
    # Filtrar datos para el país seleccionado
    events_filtered = events[events.Country == country_selected].copy()

    # Configurar el índice y ordenar por fecha
    events_filtered.set_index('DATE', inplace=True)
    events_filtered.sort_index(ascending=True, inplace=True)

    # Asegurarse de que las columnas numéricas estén en el tipo correcto
    events_filtered['GoldsteinScaleWA'] = events_filtered['GoldsteinScaleWA'].astype(float)
    events_filtered['ToneWA'] = events_filtered['ToneWA'].astype(float)

    # Crear características adicionales
    events_filtered['day'] = events_filtered.index.day
    events_filtered['week'] = events_filtered.index.isocalendar().week
    events_filtered['month'] = events_filtered.index.month
    events_filtered['year'] = events_filtered.index.year
    events_filtered['day_of_week'] = events_filtered.index.dayofweek

    events_filtered['day_sin'] = np.sin(2 * np.pi * events_filtered['day'] / 31)
    events_filtered['day_cos'] = np.cos(2 * np.pi * events_filtered['day'] / 31)
    events_filtered['week_sin'] = np.sin(2 * np.pi * events_filtered['week'] / 52)
    events_filtered['week_cos'] = np.cos(2 * np.pi * events_filtered['week'] / 52)
    events_filtered['month_sin'] = np.sin(2 * np.pi * events_filtered['month'] / 12)
    events_filtered['month_cos'] = np.cos(2 * np.pi * events_filtered['month'] / 12)
    cycle_length = 10
    events_filtered['year_sin'] = np.sin(2 * np.pi * events_filtered['year'] / cycle_length)
    events_filtered['year_cos'] = np.cos(2 * np.pi * events_filtered['year'] / cycle_length)
    events_filtered['day_of_week_sin'] = np.sin(2 * np.pi * events_filtered['day_of_week'] / 7)
    events_filtered['day_of_week_cos'] = np.cos(2 * np.pi * events_filtered['day_of_week'] / 7)

    events_filtered['GoldsteinScaleWA_lag1'] = events_filtered['GoldsteinScaleWA'].shift(1)
    events_filtered['GoldsteinScaleWA_lag7'] = events_filtered['GoldsteinScaleWA'].shift(7)
    events_filtered['GoldsteinScaleWA_lag30'] = events_filtered['GoldsteinScaleWA'].shift(30)
    events_filtered['ToneWA_lag1'] = events_filtered['ToneWA'].shift(1)

    events_filtered['GoldsteinScaleWA_roll7'] = events_filtered['GoldsteinScaleWA'].rolling(window=7).mean()
    events_filtered['ToneWA_roll7'] = events_filtered['ToneWA'].rolling(window=7).mean()

    events_filtered.dropna(inplace=True)

    # Separar las características y el target
    X = events_filtered[['day_sin', 'day_cos', 'week_sin', 'week_cos', 'month_sin', 'month_cos', 
                        'year_sin', 'year_cos', 'day_of_week_sin', 'day_of_week_cos', 
                        'GoldsteinScaleWA_lag1', 'GoldsteinScaleWA_lag7', 'GoldsteinScaleWA_lag30',
                        'ToneWA_lag1', 'GoldsteinScaleWA_roll7', 'ToneWA_roll7']]
    y = events_filtered['GoldsteinScaleWA']

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Entrenar el modelo
    best_rf = RandomForestRegressor(
        max_depth=4, 
        max_features=1.0, 
        min_samples_leaf=6,
        min_samples_split=30, 
        n_estimators=200
    )

    best_rf.fit(X_train, y_train)
    y_pred_train = best_rf.predict(X_train)
    y_pred_test = best_rf.predict(X_test)

    train_dates, test_dates = X_train.index, X_test.index

    # Guardar resultados de entrenamiento y prueba
    results_train = pd.DataFrame({
        'DATE': train_dates,
        'pais': country_selected,
        'y_pred': y_pred_train,
        'y_real': y_train
    })

    results_test = pd.DataFrame({
        'DATE': test_dates,
        'pais': country_selected,
        'y_pred': y_pred_test,
        'y_real': y_test
    })

    results = pd.concat([results_train, results_test])

    df_combined = pd.concat([df_combined, results])

    # Generar predicciones futuras
    last_date = events_filtered.index.max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]

    future_df = pd.DataFrame(index=future_dates)
    future_df['day'] = future_df.index.day
    future_df['week'] = future_df.index.isocalendar().week
    future_df['month'] = future_df.index.month
    future_df['year'] = future_df.index.year
    future_df['day_of_week'] = future_df.index.dayofweek

    future_df['day_sin'] = np.sin(2 * np.pi * future_df['day'] / 31)
    future_df['day_cos'] = np.cos(2 * np.pi * future_df['day'] / 31)
    future_df['week_sin'] = np.sin(2 * np.pi * future_df['week'] / 52)
    future_df['week_cos'] = np.cos(2 * np.pi * future_df['week'] / 52)
    future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
    future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)
    future_df['year_sin'] = np.sin(2 * np.pi * future_df['year'] / cycle_length)
    future_df['year_cos'] = np.cos(2 * np.pi * future_df['year'] / cycle_length)
    future_df['day_of_week_sin'] = np.sin(2 * np.pi * future_df['day_of_week'] / 7)
    future_df['day_of_week_cos'] = np.cos(2 * np.pi * future_df['day_of_week'] / 7)

    future_df['GoldsteinScaleWA_lag1'] = events_filtered['GoldsteinScaleWA'].shift(1).fillna(method='bfill').values[-len(future_dates):]
    future_df['GoldsteinScaleWA_lag7'] = events_filtered['GoldsteinScaleWA'].shift(7).fillna(method='bfill').values[-len(future_dates):]
    future_df['GoldsteinScaleWA_lag30'] = events_filtered['GoldsteinScaleWA'].shift(30).fillna(method='bfill').values[-len(future_dates):]
    future_df['ToneWA_lag1'] = events_filtered['ToneWA'].shift(1).fillna(method='bfill').values[-len(future_dates):]

    future_df['GoldsteinScaleWA_roll7'] = events_filtered['GoldsteinScaleWA'].rolling(window=7).mean().fillna(method='bfill').values[-len(future_dates):]
    future_df['ToneWA_roll7'] = events_filtered['ToneWA'].rolling(window=7).mean().fillna(method='bfill').values[-len(future_dates):]

    X_future = future_df[['day_sin', 'day_cos', 'week_sin', 'week_cos', 'month_sin', 'month_cos', 
                          'year_sin', 'year_cos', 'day_of_week_sin', 'day_of_week_cos', 
                          'GoldsteinScaleWA_lag1', 'GoldsteinScaleWA_lag7', 'GoldsteinScaleWA_lag30',
                          'ToneWA_lag1', 'GoldsteinScaleWA_roll7', 'ToneWA_roll7']]

    y_future_pred = best_rf.predict(X_future)

    future_results = pd.DataFrame({
        'DATE': future_dates,
        'pais': country_selected,
        'y_pred': y_future_pred,
        'y_real': np.nan
    })

    df_combined = pd.concat([df_combined, future_results])

# Guardar el DataFrame combinado en un archivo CSV
df_combined.to_csv(output_file, index=False)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StringType
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, dayofmonth, weekofyear, month, year, dayofweek, sin, cos, pi, lag, mean
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator


# COMMAND ----------

# account for landing files from https
storage_account_name = "factoredatathon2024"
storage_account_key = "wgbe0Fzs4W3dPNc35dp//uumz+SPDXVLLGu0mNaxTs2VLHCCPnD7u79PYt4mKeSFboqMRnZ+s+ez+ASty+k+sQ=="
container_name = "silver"

# Configure Spark to use the storage account key
spark.conf.set(f"fs.azure.account.key.{storage_account_name}.dfs.core.windows.net", storage_account_key)

# Define the path to write the DataFrame
file_path = f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/gkgLabeledSilver1"


# Read the Parquet files
df = spark.read.format("delta").load(file_path)

# COMMAND ----------

countries_less_than_500 = df.groupBy("countryCode").count().filter("count < 1000")

# COMMAND ----------

# Agrupar por 'countryCode', contar las filas, y ordenar de menor a mayor
country_count = df.groupBy("countryCode").count().orderBy("count", ascending=True)

# Mostrar las primeras 50 filas
country_count.show(50)

# COMMAND ----------

# Agrupar por 'countryCode' y contar el número de filas por país
country_counts = df.groupBy("countryCode").count()

# Filtrar los países que tienen 1000 o más noticias
countries_to_keep = country_counts.filter(F.col("count") >= 1000).select("countryCode")

# Filtrar el DataFrame original para mantener solo las filas de esos países
df = df.join(countries_to_keep, on="countryCode", how="inner")

# COMMAND ----------

# Contar el total de etiquetas en Cluster_Name2
df = df.withColumn("Total_Labels", F.size(F.col("Cluster_Name2")))

# Crear columnas para cada etiqueta y calcular el porcentaje
for label in ["SOCIAL", "POLITICAL", "ECONOMIC"]:
    df = df.withColumn(
        f"{label}_Percentage",
        (F.size(F.expr(f"filter(Cluster_Name2, x -> x = '{label}')")) / F.col("Total_Labels")) * 100
    )

# Eliminar la columna Total_Labels si no la necesitas
df = df.drop("Total_Labels")

# COMMAND ----------

df_pd = df.select("SOCIAL_Percentage", "POLITICAL_Percentage", "ECONOMIC_Percentage").toPandas()

# Calcular el promedio de los porcentajes para cada etiqueta
avg_percentages = df_pd.mean()

import matplotlib.pyplot as plt

# Crear un gráfico de barras para los promedios
avg_percentages.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'])

# Personalizar el gráfico
plt.title("Promedio de Porcentajes por Etiqueta")
plt.xlabel("Etiqueta")
plt.ylabel("Porcentaje (%)")
plt.xticks(rotation=0)

# Mostrar el gráfico
plt.show()

# COMMAND ----------

# Crear la columna TONE_AVG_ECONOMIC
df = df.withColumn("TONE_AVG_ECONOMIC", F.col("ECONOMIC_Percentage") * F.col("TONE_AVG") / 100)

# Mostrar el resultado
df.select("TONE_AVG", "ECONOMIC_Percentage", "TONE_AVG_ECONOMIC").show(truncate=False)

# COMMAND ----------

df.columns

# COMMAND ----------

# MAGIC %md ## Modularized Code:
# MAGIC

# COMMAND ----------

# %pip install sqlalchemy

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col, to_date, dayofmonth, weekofyear, month, year, dayofweek, lag
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
# from sqlalchemy import create_engine


# 1. Función para la conexión con Azure y Spark
def load_data_from_azure(storage_account_name, storage_account_key, container_name, spark):
    try:
        # Configurar Spark para usar la clave de la cuenta de almacenamiento
        spark.conf.set(f"fs.azure.account.key.{storage_account_name}.dfs.core.windows.net", storage_account_key)

        # Definir la ruta para leer los archivos Parquet
        file_path = f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/gkgLabeledSilver1"

        # Leer los archivos Parquet
        df = spark.read.format("delta").load(file_path)
        df.show(5)
        return df
    except Exception as e:
        print(f"Error loading data from Azure: {e}")
        return None

# 2. Función para contar el total de etiquetas y calcular porcentajes
def calculate_label_percentages(df):
    df = df.withColumn("Total_Labels", F.size(F.col("Cluster_Name2")))

    for label in ["SOCIAL", "POLITICAL", "ECONOMIC"]:
        df = df.withColumn(
            f"{label}_Percentage",
            (F.size(F.expr(f"filter(Cluster_Name2, x -> x = '{label}')")) / F.col("Total_Labels")) * 100
        )

    df = df.withColumn("TONE_AVG_ECONOMIC", F.col("ECONOMIC_Percentage") * F.col("TONE_AVG") / 100)
    df = df.drop("Total_Labels")

    return df

# 3. Función para preparar y filtrar los eventos por país
def filter_and_prepare_events(events, country_selected):
    events = events.withColumnRenamed("date0", "DATE").withColumnRenamed("countryCode", "Country")
    events = events.filter(F.col("Country") == country_selected)

    # Convertir la columna DATE a formato de fecha y agregar TONE_AVG_ECONOMIC
    events = events.withColumn("DATE", F.to_date(F.col("DATE"), "yyyy-MM-dd"))\
                   .withColumn("TONE_AVG_ECONOMIC", F.col("TONE_AVG_ECONOMIC").cast("float"))

    # Agrupar por DATE y Country, y calcular el promedio de TONE_AVG_ECONOMIC
    events_grouped = events.groupBy("DATE", "Country").agg(F.avg("TONE_AVG_ECONOMIC").alias("TONE_AVG_ECONOMIC"))

    return events_grouped.sort("DATE")

# 4. Función para la ingeniería de características
def feature_engineering(events_filtered):
    events_filtered = events_filtered.withColumn("day", dayofmonth(col("DATE")))\
                                     .withColumn("week", weekofyear(col("DATE")))\
                                     .withColumn("month", month(col("DATE")))\
                                     .withColumn("year", year(col("DATE")))\
                                     .withColumn("day_of_week", dayofweek(col("DATE")))

    # Definir pi como una constante en el contexto de PySpark
    pi_value = 3.141592653589793
    
    events_filtered = events_filtered.withColumn("day_sin", F.sin(2 * pi_value * col("day") / 31))\
                                     .withColumn("day_cos", F.cos(2 * pi_value * col("day") / 31))\
                                     .withColumn("week_sin", F.sin(2 * pi_value * col("week") / 52))\
                                     .withColumn("week_cos", F.cos(2 * pi_value * col("week") / 52))\
                                     .withColumn("month_sin", F.sin(2 * pi_value * col("month") / 12))\
                                     .withColumn("month_cos", F.cos(2 * pi_value * col("month") / 12))\
                                     .withColumn("year_sin", F.sin(2 * pi_value * col("year") / 10))\
                                     .withColumn("year_cos", F.cos(2 * pi_value * col("year") / 10))\
                                     .withColumn("day_of_week_sin", F.sin(2 * pi_value * col("day_of_week") / 7))\
                                     .withColumn("day_of_week_cos", F.cos(2 * pi_value * col("day_of_week") / 7))

    window_spec = Window.orderBy("DATE")

    events_filtered = events_filtered.withColumn("TONE_AVG_ECONOMIC_lag1", lag(col("TONE_AVG_ECONOMIC"), 1).over(window_spec))\
                                     .withColumn("TONE_AVG_ECONOMIC_lag7", lag(col("TONE_AVG_ECONOMIC"), 7).over(window_spec))\
                                     .withColumn("TONE_AVG_ECONOMIC_lag30", lag(col("TONE_AVG_ECONOMIC"), 30).over(window_spec))

    events_filtered = events_filtered.withColumn("TONE_AVG_ECONOMIC_roll7", F.mean(col("TONE_AVG_ECONOMIC")).over(Window.rowsBetween(-6, 0)))

    return events_filtered.na.drop()

# 5. Función para entrenar y evaluar el modelo
def train_model_and_get_predictions(events_filtered):
    feature_columns = ['day_sin', 'day_cos', 'week_sin', 'week_cos', 'month_sin', 'month_cos', 
                       'year_sin', 'year_cos', 'day_of_week_sin', 'day_of_week_cos', 
                       'TONE_AVG_ECONOMIC_lag1', 'TONE_AVG_ECONOMIC_lag7', 'TONE_AVG_ECONOMIC_lag30', 
                       'TONE_AVG_ECONOMIC_roll7']
    print(f"Number of rows before assembling features: {events_filtered.count()}")
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    assembled_data = assembler.transform(events_filtered)

    if assembled_data.count() == 0:
        raise ValueError("Assembled data is empty. Cannot train model.")

    train_data, test_data = assembled_data.randomSplit([0.8, 0.2], seed=42)

    if train_data.count() == 0 or test_data.count() == 0:
        raise ValueError("Training or testing data is empty. Cannot train model.")

    rf = RandomForestRegressor(featuresCol="features", labelCol="TONE_AVG_ECONOMIC", maxDepth=4, numTrees=200)
    model = rf.fit(train_data)

    predictions_train = model.transform(train_data)
    predictions_test = model.transform(test_data)

    evaluator = RegressionEvaluator(labelCol="TONE_AVG_ECONOMIC", predictionCol="prediction", metricName="rmse")

    rmse_train = evaluator.evaluate(predictions_train)
    rmse_test = evaluator.evaluate(predictions_test)

    return predictions_train, predictions_test, rmse_train, rmse_test

# 5. Función para guardar los resultados en un CSV
"""
def save_results_to_csv(predictions_train, predictions_test, output_file):
    train_results = predictions_train.select(col("DATE"), col("Country"), col("prediction").alias("y_pred"), col("TONE_AVG_ECONOMIC").alias("y_real"))
    test_results = predictions_test.select(col("DATE"), col("Country"), col("prediction").alias("y_pred"), col("TONE_AVG_ECONOMIC").alias("y_real"))

    results = train_results.union(test_results)

    results.write.mode("append").option("header", "true").csv(output_file)
"""


def save_results_to_sql(predictions_train, predictions_test, jdbc_url, table_name, connection_properties):
    from pyspark.sql import DataFrame
    from pyspark.sql.functions import col

    # Selecciona las columnas necesarias de los conjuntos de datos de entrenamiento y prueba
    train_results = predictions_train.select(
        col("DATE"),
        col("Country"),
        col("prediction").alias("y_pred"),
        col("TONE_AVG_ECONOMIC").alias("y_real")
    )
    test_results = predictions_test.select(
        col("DATE"),
        col("Country"),
        col("prediction").alias("y_pred"),
        col("TONE_AVG_ECONOMIC").alias("y_real")
    )

    # Combina los resultados de entrenamiento y prueba
    results = train_results.union(test_results)

    # Guarda los resultados en la base de datos SQL
    results.write.mode("append") \
        .jdbc(url=jdbc_url, table=table_name, properties=connection_properties)

# 6. Función principal
def main():

    spark = SparkSession.builder.appName("AzureDataAnalysis").getOrCreate()
    storage_account_name = "factoredatathon2024"
    storage_account_key = "wgbe0Fzs4W3dPNc35dp//uumz+SPDXVLLGu0mNaxTs2VLHCCPnD7u79PYt4mKeSFboqMRnZ+s+ez+ASty+k+sQ=="
    container_name = "silver"
    output_file = "gkg_model_predictions.csv"

    df = load_data_from_azure(storage_account_name, storage_account_key, container_name, spark)
    
    
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

    # Define the target table name
    table_name = "gkg.tonePredictionsGold"


    if df is None:
        raise Exception("DataFrame is None. Exiting the process.")
    
    df = calculate_label_percentages(df)
    min_rows_for_training = 100  # Define the minimum number of rows required for training

    countries = df.select("countryCode").distinct().rdd.flatMap(lambda x: x).collect()

    for country in countries:
        print(f"Processing country: {country}")
        events_filtered = filter_and_prepare_events(df, country)

        if events_filtered is not None and events_filtered.count() > min_rows_for_training:
            events_filtered = feature_engineering(events_filtered)

            predictions_train, predictions_test, rmse_train, rmse_test = train_model_and_get_predictions(events_filtered)

            print(f"RMSE Train: {rmse_train}, RMSE Test: {rmse_test}")

            save_results_to_sql(predictions_train, predictions_test, jdbc_url, table_name, connection_properties)
        else:
            print(f"Not enough data for country {country}. Skipping...")

    print(f"Results saved to {output_file}")


main()


# COMMAND ----------

# Leer el archivo CSV desde DBFS
df = spark.read.format("csv").option("header", "true").load("dbfs:/gkg_model_predictions.csv")

# Mostrar las primeras filas del DataFrame
df.show(5)

# COMMAND ----------

# Leer el archivo CSV desde DBFS
df = spark.read.format("csv").option("header", "true").load("dbfs:/gkg_model_predictions.csv")

# Convertir las columnas a los tipos de datos adecuados
df = df.withColumn("DATE", F.to_date(F.col("DATE"), "yyyy-MM-dd"))\
       .withColumn("y_pred", F.col("y_pred").cast("float"))\
       .withColumn("y_real", F.col("y_real").cast("float"))

# Filtrar por país (por ejemplo, "US")
country_df = df.filter(F.col("Country") == "BR")

pandas_df = country_df.toPandas()

pandas_df = pandas_df.sort_values(by='DATE')

import matplotlib.pyplot as plt

# Crear un gráfico de líneas
plt.figure(figsize=(14, 7))

# Graficar y_pred
plt.plot(pandas_df['DATE'], pandas_df['y_pred'], label='Predicción', color='red', linestyle="--")

# Graficar y_real
plt.plot(pandas_df['DATE'], pandas_df['y_real'], label='Valor Real', color='blue')

# Añadir títulos y etiquetas
plt.title('Predicción vs Valor Real')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)

# Mostrar gráfico
plt.show()

# COMMAND ----------

output = pd.read_csv('gkg_model_predictions.csv')

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# 1. Filtrar los datos para Estados Unidos (US)
us_events = df.filter(col("countryCode") == "US")

# 2. Convertir a Pandas DataFrame
us_events_pd = us_events.toPandas()

# 3. Graficar
plt.figure(figsize=(14, 7))

# Graficar TONE_AVG_ECONOMIC original
plt.plot(us_events_pd['DATE'], us_events_pd['TONE_AVG_ECONOMIC'], label='TONE_AVG_ECONOMIC', color='blue')

# Graficar TONE_AVG_ECONOMIC con lag
plt.plot(us_events_pd['DATE'], us_events_pd['TONE_AVG_ECONOMIC_lag1'], label='TONE_AVG_ECONOMIC_lag1', color='red', linestyle='--')
plt.plot(us_events_pd['DATE'], us_events_pd['TONE_AVG_ECONOMIC_lag7'], label='TONE_AVG_ECONOMIC_lag7', color='green', linestyle='--')
plt.plot(us_events_pd['DATE'], us_events_pd['TONE_AVG_ECONOMIC_lag30'], label='TONE_AVG_ECONOMIC_lag30', color='orange', linestyle='--')

# Configuración de la gráfica
plt.title('TONE_AVG_ECONOMIC y sus Lags en Estados Unidos (US)')
plt.xlabel('Fecha')
plt.ylabel('TONE_AVG_ECONOMIC')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Mostrar la gráfica
plt.show()

# COMMAND ----------

# MAGIC %md ## Intento Fallido:
# MAGIC

# COMMAND ----------

pandas_df = df.select('date0',
                        'countryCode',
                        'TONE_AVG',
                        'THEMES2',
                        'SOCIAL_Percentage',
                        'POLITICAL_Percentage',
                        'ECONOMIC_Percentage',
                        'TONE_AVG_ECONOMIC').toPandas()

# COMMAND ----------

pandas_df.head()

# COMMAND ----------

country_selected = 'US'  # Reemplaza 'CH' con el nombre del país que deseas filtrar
df_filtered = pandas_df[pandas_df.countryCode == country_selected].copy()

# COMMAND ----------

df_filtered['date0'] = pd.to_datetime(df_filtered['date0'])
df_filtered.set_index('date0', inplace=True)

# COMMAND ----------

display(events)print(df_model.columns)

# Mostrar un ejemplo de los datos
df_model.select("TONE_AVG", "SOCIAL_Percentage", "POLITICAL_Percentage").show(truncate=False)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

from pyspark.sql.types import DoubleType


# 1. Preparación de los Datos
df_model = df.select("TONE_AVG","SOCIAL_Percentage", "POLITICAL_Percentage", "TONE_AVG_ECONOMIC")

df_model = df_model.withColumn("TONE_AVG", F.col("TONE_AVG").cast(DoubleType()))
df_model = df_model.withColumn("SOCIAL_Percentage", F.col("SOCIAL_Percentage").cast(DoubleType()))
df_model = df_model.withColumn("POLITICAL_Percentage", F.col("POLITICAL_Percentage").cast(DoubleType()))


# 2. VectorAssembler para características
assembler = VectorAssembler(inputCols=["TONE_AVG","SOCIAL_Percentage", "POLITICAL_Percentage"], outputCol="features")
df_model = assembler.transform(df_model)

# 3. Definir la variable objetivo
df_model = df_model.withColumnRenamed("TONE_AVG_ECONOMIC", "label")

# 4. División de los Datos
train_data, test_data = df_model.randomSplit([0.8, 0.2], seed=42)

# 5. Entrenamiento del Modelo
rfr = RandomForestRegressor(featuresCol="features", labelCol="label")
model = rfr.fit(train_data)

# 6. Predicción en el conjunto de prueba
predictions = model.transform(test_data)

# 7. Evaluación del Modelo
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"RMSE en el conjunto de prueba: {rmse}")

# Mostrar algunas predicciones
predictions.select("features", "label", "prediction").show(10)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, DateType, StringType

# Asegurarte de que 'date0' es de tipo fecha y 'countryCode' es de tipo string
df_model = df_model.withColumn("date0", F.col("date0").cast(DateType()))
df_model = df_model.withColumn("countryCode", F.col("countryCode").cast(StringType()))

# Asegurar que las columnas de características son numéricas
df_model = df_model.withColumn("TONE_AVG", F.col("TONE_AVG").cast(DoubleType()))
df_model = df_model.withColumn("SOCIAL_Percentage", F.col("SOCIAL_Percentage").cast(DoubleType()))
df_model = df_model.withColumn("POLITICAL_Percentage", F.col("POLITICAL_Percentage").cast(DoubleType()))

# Crear la columna features combinando TONE_AVG, SOCIAL_Percentage y POLITICAL_Percentage
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=["TONE_AVG", "SOCIAL_Percentage", "POLITICAL_Percentage"], outputCol="features")
df_model = assembler.transform(df_model)

# Mostrar las primeras filas para verificar
df_model.select("date0", "countryCode", "TONE_AVG", "SOCIAL_Percentage", "POLITICAL_Percentage", "features").show(truncate=False)

# 3. Definir la variable objetivo
df_model = df_model.withColumnRenamed("TONE_AVG_ECONOMIC", "label")

# 4. División de los Datos
train_data, test_data = df_model.randomSplit([0.8, 0.2], seed=42)

# 5. Entrenamiento del Modelo
rfr = RandomForestRegressor(featuresCol="features", labelCol="label")
model = rfr.fit(train_data)

# 6. Predicción en el conjunto de prueba
predictions = model.transform(test_data)

# 7. Evaluación del Modelo
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"RMSE en el conjunto de prueba: {rmse}")

# Mostrar algunas predicciones
predictions.select("features", "label", "prediction").show(10)


# COMMAND ----------

predictions.columns


# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
import seaborn as sns

# Convertir a Pandas DataFrame
predictions_pd = predictions.select("date0", "countryCode", "label", "prediction").toPandas()

# Agrupar por país y fecha
grouped_predictions = predictions_pd.groupby(["countryCode", "date0"]).reset_index()

# Filtrar datos para un país específico, por ejemplo 'US'
country_data = grouped_predictions[grouped_predictions["countryCode"] == "US"]

# Graficar
plt.figure(figsize=(12, 6))
sns.lineplot(data=country_data, x="date0", y="prediction", marker='o')
plt.title("Predicciones del Tono Económico en el Tiempo para el País US")
plt.xlabel("Fecha")
plt.ylabel("Predicción del Tono Económico")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
