# Databricks notebook source
# MAGIC %md #Time Series Model
# MAGIC

# COMMAND ----------

# MAGIC %md ## 1. Connection with Azure and Spark (DELETE KEYS!)
# MAGIC

# COMMAND ----------

storage_account_key = "wgbe0Fzs4W3dPNc35dp//uumz+SPDXVLLGu0mNaxTs2VLHCCPnD7u79PYt4mKeSFboqMRnZ+s+ez+ASty+k+sQ=="
storage_account_name = "factoredatathon2024"
container_name = "gold"

spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net",
    f"{storage_account_key}"
)

file_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/weightedAvgGoldsteinToneGold.csv"
df = spark.read.format("csv").option("header", "false").load(file_path)

# COMMAND ----------

# Install the SQLAlchemy
%pip install sqlalchemy

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sqlalchemy import create_engine
df_combined = pd.DataFrame()

# 1. Function for connecting to Azure and Spark
def load_data_from_azure(storage_account_name, container_name, file_name, spark):
    file_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/{file_name}"
    df = spark.read.format("csv").option("header", "false").load(file_path)
    
    # Show the first 5 rows
    df.show(5)
    return df

# 2. Function to prepare and filter events by country
def filter_and_prepare_events(events, country_selected):
    column_names = ["DATE", "Country", "GoldsteinScaleWA", "ToneWA"]
    events = events.toDF(*column_names).toPandas()

    if 'Country' not in events.columns:
        print("Error: La columna 'Country' no se encontró en el DataFrame después de renombrar las columnas.")
        return None

    # Filter by country
    events_filtered = events[events.Country == country_selected].copy()

    # "Date" to Datetime and set it as index
    events_filtered['DATE'] = pd.to_datetime(events_filtered['DATE'])
    events_filtered.set_index('DATE', inplace=True)
    events_filtered.sort_index(ascending=True, inplace=True)

    # Cast to float
    events_filtered['GoldsteinScaleWA'] = events_filtered['GoldsteinScaleWA'].astype(float)
    events_filtered['ToneWA'] = events_filtered['ToneWA'].astype(float)
    
    return events_filtered

# 3. Function to create features
def feature_engineering(events_filtered):
   
    events_filtered['day'] = events_filtered.index.day
    events_filtered['week'] = events_filtered.index.isocalendar().week
    events_filtered['month'] = events_filtered.index.month
    events_filtered['year'] = events_filtered.index.year
    events_filtered['day_of_week'] = events_filtered.index.dayofweek

    # Cyclic features
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

    # Lag features
    events_filtered['GoldsteinScaleWA_lag1'] = events_filtered['GoldsteinScaleWA'].shift(1)
    events_filtered['GoldsteinScaleWA_lag7'] = events_filtered['GoldsteinScaleWA'].shift(7)
    events_filtered['GoldsteinScaleWA_lag30'] = events_filtered['GoldsteinScaleWA'].shift(30)
    events_filtered['ToneWA_lag1'] = events_filtered['ToneWA'].shift(1)

    # Rolling windows
    events_filtered['GoldsteinScaleWA_roll7'] = events_filtered['GoldsteinScaleWA'].rolling(window=7).mean()
    events_filtered['ToneWA_roll7'] = events_filtered['ToneWA'].rolling(window=7).mean()

    events_filtered.dropna(inplace=True)

    return events_filtered

# 4. Function to train and evaluate the model (adjusted to handle cases with limited samples)
def train_model_and_get_predictions(events_filtered):
    X = events_filtered[['day_sin', 'day_cos', 'week_sin', 'week_cos', 'month_sin', 'month_cos', 
                         'year_sin', 'year_cos', 'day_of_week_sin', 'day_of_week_cos', 
                         'GoldsteinScaleWA_lag1', 'GoldsteinScaleWA_lag7', 'GoldsteinScaleWA_lag30', 
                         'ToneWA_lag1', 'GoldsteinScaleWA_roll7', 'ToneWA_roll7']]

    y = events_filtered['GoldsteinScaleWA']

    if len(X) < 2:
        print("Error: No hay suficientes muestras para entrenar el modelo para este país.")
        return None, None, None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

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

    return X_train.index, X_test.index, y_pred_train, y_pred_test, y_train, y_test

# 5. Function to save results to SQL (adjusted to include y_train and y_test)
def save_results_to_sql(train_dates, test_dates, y_pred_train, y_pred_test, y_train, y_test, country, output_file):
    # Create train DF
    results_train = pd.DataFrame({
        'fecha': train_dates,
        'pais': country,
        'y_pred': y_pred_train,
        'y_real': y_train
    })

    # Create test DF
    results_test = pd.DataFrame({
        'fecha': test_dates,
        'pais': country,
        'y_pred': y_pred_test,
        'y_real': y_test
    })

    # Connect the results
    results = pd.concat([results_train, results_test])

    global df_combined

    df_combined = pd.concat([df_combined, results])

    # Write as SQL table
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

    # Write the DataFrame to SQL Server
    spark_result = spark.createDataFrame(df_combined)

    # Define the target table name
    table_name = "events.goldsteinPredictionsGold"

    # Write the Spark DataFrame to Azure SQL Database
    spark_result.write \
        .jdbc(url=jdbc_url, table=table_name, mode='overwrite', properties=connection_properties)


# 6. Main function
def main():
    storage_account_name = "factoredatathon2024"
    container_name = "gold"
    file_name = "weightedAvgGoldsteinToneGold.csv"
    output_file = "model_predictions.csv"

    storage_account_key = "wgbe0Fzs4W3dPNc35dp//uumz+SPDXVLLGu0mNaxTs2VLHCCPnD7u79PYt4mKeSFboqMRnZ+s+ez+ASty+k+sQ=="
    storage_account_name = "factoredatathon2024"
    container_name = "gold"

    spark.conf.set(
        f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net",
        f"{storage_account_key}"
    )

    file_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/weightedAvgGoldsteinToneGold.csv"
    df = spark.read.format("csv").option("header", "false").load(file_path)

    # Connection and data load
    df = load_data_from_azure(storage_account_name, container_name, file_name, spark)
    
    # Countries list
    countries = df.select('_c1').distinct().rdd.flatMap(lambda x: x).collect()


    for country in countries:
        print(f"Processing country: {country}")
        # Event preparation and filtering
        events_filtered = filter_and_prepare_events(df, country)
        
        if events_filtered is not None and len(events_filtered) > 0:
            # Feature engineering
            events_filtered = feature_engineering(events_filtered)
            
            # Model training and predictions
            train_dates, test_dates, y_pred_train, y_pred_test, y_train, y_test = train_model_and_get_predictions(events_filtered)
            
            if train_dates is not None:
                # Save results in SQL
                save_results_to_sql(train_dates, test_dates, y_pred_train, y_pred_test, y_train, y_test, country, output_file)


# Run script
if __name__ == "__main__":
    main()

# COMMAND ----------


