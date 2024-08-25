# Databricks notebook source
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

# MAGIC %md ## Modularized Code:
# MAGIC

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

