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

df.show()

# COMMAND ----------

df.select("countryCode").distinct().count()

# COMMAND ----------

countries_less_than_500 = df.groupBy("countryCode").count().filter("count < 1000")

# Count the total number of countries with fewer than 500 news articles
countries_less_than_500.count()

# COMMAND ----------

# Group by 'countryCode', count the rows, and sort in ascending order
country_count = df.groupBy("countryCode").count().orderBy("count", ascending=True)

# Show the first 50 rows
country_count.show(50)

# COMMAND ----------

# Group by 'countryCode' and count the number of rows per country
country_counts = df.groupBy("countryCode").count()

# Filter the countries that have 1000 or more news articles
countries_to_keep = country_counts.filter(F.col("count") >= 1000).select("countryCode")

# Filter the original DataFrame to keep only rows from those countries
df = df.join(countries_to_keep, on="countryCode", how="inner")

# Show the first rows of the resulting DataFrame
df.show()


# COMMAND ----------

# Count the total number of labels in Cluster_Name2
df = df.withColumn("Total_Labels", F.size(F.col("Cluster_Name2")))

# Create columns for each label and calculate the percentage
for label in ["SOCIAL", "POLITICAL", "ECONOMIC"]:
    df = df.withColumn(
        f"{label}_Percentage",
        (F.size(F.expr(f"filter(Cluster_Name2, x -> x = '{label}')")) / F.col("Total_Labels")) * 100
    )

# Drop the Total_Labels column if it's not needed
df = df.drop("Total_Labels")

# Show the result
df.show(truncate=False)


# COMMAND ----------

df_pd = df.select("SOCIAL_Percentage", "POLITICAL_Percentage", "ECONOMIC_Percentage").toPandas()

# Calculate the average percentages for each label
avg_percentages = df_pd.mean()

import matplotlib.pyplot as plt

# Create a bar chart for the averages
avg_percentages.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'])

plt.title("Average Percentages by Label")
plt.xlabel("Label")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=0)

# Show the chart
plt.show()


# COMMAND ----------

# Create TONE_AVG_ECONOMIC column
df = df.withColumn("TONE_AVG_ECONOMIC", F.col("ECONOMIC_Percentage") * F.col("TONE_AVG") / 100)

# show results()
df.select("TONE_AVG", "ECONOMIC_Percentage", "TONE_AVG_ECONOMIC").show(truncate=False)

# COMMAND ----------

df.columns

# COMMAND ----------

# MAGIC %md ## Modularized Code:
# MAGIC

# COMMAND ----------

# MAGIC %pip install sqlalchemy

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col, to_date, dayofmonth, weekofyear, month, year, dayofweek, lag
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
#from sqlalchemy import create_engine

# 1. Function for connecting to Azure and Spark
def load_data_from_azure(storage_account_name, storage_account_key, container_name, spark):
    try:
        # Configure Spark to use the storage account key
        spark.conf.set(f"fs.azure.account.key.{storage_account_name}.dfs.core.windows.net", storage_account_key)

        # Define the path to read the Parquet files
        file_path = f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/gkgLabeledSilver1"

        # Read the Parquet files
        df = spark.read.format("delta").load(file_path)
        df.show(5)
        return df
    except Exception as e:
        print(f"Error loading data from Azure: {e}")
        return None

# 2. Function to count the total number of labels and calculate percentages
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

# 3. Function to filter and prepare events by country
def filter_and_prepare_events(events, country_selected):
    events = events.withColumnRenamed("date0", "DATE").withColumnRenamed("countryCode", "Country")
    events = events.filter(F.col("Country") == country_selected)

    # Convert the DATE column to date format and add TONE_AVG_ECONOMIC
    events = events.withColumn("DATE", F.to_date(F.col("DATE"), "yyyy-MM-dd"))\
                   .withColumn("TONE_AVG_ECONOMIC", F.col("TONE_AVG_ECONOMIC").cast("float"))

    # Group by DATE and Country, and calculate the average of TONE_AVG_ECONOMIC
    events_grouped = events.groupBy("DATE", "Country").agg(F.avg("TONE_AVG_ECONOMIC").alias("TONE_AVG_ECONOMIC"))

    return events_grouped.sort("DATE")

# 4. Function for feature engineering
def feature_engineering(events_filtered):
    events_filtered = events_filtered.withColumn("day", dayofmonth(col("DATE")))\
                                     .withColumn("week", weekofyear(col("DATE")))\
                                     .withColumn("month", month(col("DATE")))\
                                     .withColumn("year", year(col("DATE")))\
                                     .withColumn("day_of_week", dayofweek(col("DATE")))

    # Define pi as a constant in the context of PySpark
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

# 5. Function to train and evaluate the model
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

# 5. Function to save results to a CSV
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

    # Select the required columns from training and testing datasets
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

    # Combine training and testing results
    results = train_results.union(test_results)

    # Save results to the SQL database
    results.write.mode("append") \
        .jdbc(url=jdbc_url, table=table_name, properties=connection_properties)

# 6. Main function
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



# COMMAND ----------

output_file

# COMMAND ----------

# MAGIC %fs ls /

# COMMAND ----------

# Read the CSV file from DBFS
df = spark.read.format("csv").option("header", "true").load("dbfs:/gkg_model_predictions.csv")

# Show the first rows
df.show(5)


# COMMAND ----------

# Read the CSV file from DBFS
df = spark.read.format("csv").option("header", "true").load("dbfs:/gkg_model_predictions.csv")

# Convert columns to the appropriate data types
df = df.withColumn("DATE", F.to_date(F.col("DATE"), "yyyy-MM-dd"))\
       .withColumn("y_pred", F.col("y_pred").cast("float"))\
       .withColumn("y_real", F.col("y_real").cast("float"))

# Filter by country (e.g., "BR")
country_df = df.filter(F.col("Country") == "BR")

pandas_df = country_df.toPandas()

pandas_df = pandas_df.sort_values(by='DATE')

import matplotlib.pyplot as plt

# Create a line plot
plt.figure(figsize=(14, 7))

# Plot y_pred
plt.plot(pandas_df['DATE'], pandas_df['y_pred'], label='Prediction', color='red', linestyle="--")

# Plot y_real
plt.plot(pandas_df['DATE'], pandas_df['y_real'], label='Actual Value', color='blue')

# Add titles and labels
plt.title('Prediction vs Actual Value')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Show plot
plt.show()


# COMMAND ----------

output = pd.read_csv('gkg_model_predictions.csv')

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# 1. Filter data for the United States (US)
us_events = df.filter(col("countryCode") == "US")

# 2. Convert to Pandas DataFrame
us_events_pd = us_events.toPandas()

# 3. Plot
plt.figure(figsize=(14, 7))

# Plot original TONE_AVG_ECONOMIC
plt.plot(us_events_pd['DATE'], us_events_pd['TONE_AVG_ECONOMIC'], label='TONE_AVG_ECONOMIC', color='blue')

# Plot TONE_AVG_ECONOMIC with lags
plt.plot(us_events_pd['DATE'], us_events_pd['TONE_AVG_ECONOMIC_lag1'], label='TONE_AVG_ECONOMIC_lag1', color='red', linestyle='--')
plt.plot(us_events_pd['DATE'], us_events_pd['TONE_AVG_ECONOMIC_lag7'], label='TONE_AVG_ECONOMIC_lag7', color='green', linestyle='--')
plt.plot(us_events_pd['DATE'], us_events_pd['TONE_AVG_ECONOMIC_lag30'], label='TONE_AVG_ECONOMIC_lag30', color='orange', linestyle='--')

# Graph settings
plt.title('TONE_AVG_ECONOMIC and its Lags in the United States (US)')
plt.xlabel('Date')
plt.ylabel('TONE_AVG_ECONOMIC')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
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

country_selected = 'US'  # Replace 'CH' with the name of the country you want to filter
df_filtered = pandas_df[pandas_df.countryCode == country_selected].copy()

# COMMAND ----------

df_filtered['date0'] = pd.to_datetime(df_filtered['date0'])
df_filtered.set_index('date0', inplace=True)

# COMMAND ----------

display(events)print(df_model.columns)

# Show an example of the data
df_model.select("TONE_AVG", "SOCIAL_Percentage", "POLITICAL_Percentage").show(truncate=False)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

from pyspark.sql.types import DoubleType


# 1. Data preparation
df_model = df.select("TONE_AVG","SOCIAL_Percentage", "POLITICAL_Percentage", "TONE_AVG_ECONOMIC")

df_model = df_model.withColumn("TONE_AVG", F.col("TONE_AVG").cast(DoubleType()))
df_model = df_model.withColumn("SOCIAL_Percentage", F.col("SOCIAL_Percentage").cast(DoubleType()))
df_model = df_model.withColumn("POLITICAL_Percentage", F.col("POLITICAL_Percentage").cast(DoubleType()))

# 2. VectorAssembler for features
assembler = VectorAssembler(inputCols=["TONE_AVG","SOCIAL_Percentage", "POLITICAL_Percentage"], outputCol="features")
df_model = assembler.transform(df_model)

# 3. Define the target variable
df_model = df_model.withColumnRenamed("TONE_AVG_ECONOMIC", "label")

# 4. Data Splitting
train_data, test_data = df_model.randomSplit([0.8, 0.2], seed=42)

# 5. Model Training
rfr = RandomForestRegressor(featuresCol="features", labelCol="label")
model = rfr.fit(train_data)

# 6. Prediction on the test set
predictions = model.transform(test_data)

# 7. Model Evaluation
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"RMSE on the test set: {rmse}")

# Show some predictions
predictions.select("features", "label", "prediction").show(10)


# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, DateType, StringType

# Ensure 'date0' is of type date and 'countryCode' is of type string
df_model = df_model.withColumn("date0", F.col("date0").cast(DateType()))
df_model = df_model.withColumn("countryCode", F.col("countryCode").cast(StringType()))

# Ensure feature columns are numeric
df_model = df_model.withColumn("TONE_AVG", F.col("TONE_AVG").cast(DoubleType()))
df_model = df_model.withColumn("SOCIAL_Percentage", F.col("SOCIAL_Percentage").cast(DoubleType()))
df_model = df_model.withColumn("POLITICAL_Percentage", F.col("POLITICAL_Percentage").cast(DoubleType()))

# Create the features column combining TONE_AVG, SOCIAL_Percentage, and POLITICAL_Percentage
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=["TONE_AVG", "SOCIAL_Percentage", "POLITICAL_Percentage"], outputCol="features")
df_model = assembler.transform(df_model)

# Show the first rows for verification
df_model.select("date0", "countryCode", "TONE_AVG", "SOCIAL_Percentage", "POLITICAL_Percentage", "features").show(truncate=False)

# 3. Define the target variable
df_model = df_model.withColumnRenamed("TONE_AVG_ECONOMIC", "label")

# 4. Data Splitting
train_data, test_data = df_model.randomSplit([0.8, 0.2], seed=42)

# 5. Model Training
rfr = RandomForestRegressor(featuresCol="features", labelCol="label")
model = rfr.fit(train_data)

# 6. Prediction on the test set
predictions = model.transform(test_data)

# 7. Model Evaluation
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"RMSE on the test set: {rmse}")

# Show some predictions
predictions.select("features", "label", "prediction").show(10)



# COMMAND ----------

predictions.columns


# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
import seaborn as sns

# Convert to Pandas DataFrame
predictions_pd = predictions.select("date0", "countryCode", "label", "prediction").toPandas()

# Group by country and date
grouped_predictions = predictions_pd.groupby(["countryCode", "date0"]).reset_index()

# Filter data for a specific country, e.g., 'US'
country_data = grouped_predictions[grouped_predictions["countryCode"] == "US"]

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=country_data, x="date0", y="prediction", marker='o')
plt.title("Economic Tone Predictions Over Time for Country US")
plt.xlabel("Date")
plt.ylabel("Economic Tone Prediction")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# COMMAND ----------


