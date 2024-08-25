# Databricks notebook source
# MAGIC %md #Time Series Model
# MAGIC

# COMMAND ----------

# MAGIC %md ## 1. Connection with Azure and Spark (DELETE KEYS!)
# MAGIC

# COMMAND ----------

storage_account_key = dbutils.secrets.get(scope="events", key="DataLakeKey")
storage_account_name = "factoredatathon2024"
container_name = "gold"

spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net",
    f"{storage_account_key}"
)

file_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/weightedAvgGoldsteinToneGold.csv"
df = spark.read.format("csv").option("header", "false").load(file_path)

# COMMAND ----------

# MAGIC %md ## 2.Library Imports:
# MAGIC

# COMMAND ----------

# MAGIC %pip install xgboost
# MAGIC      
# MAGIC from sklearn.model_selection import train_test_split
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC import matplotlib.pyplot as plt
# MAGIC import xgboost as xgb
# MAGIC from sklearn.model_selection import GridSearchCV
# MAGIC from statsmodels.tsa.arima.model import ARIMA
# MAGIC
# MAGIC

# COMMAND ----------

column_names = ["DATE", "Country", "GoldsteinScaleWA","ToneWA"]
df = df.toDF(*column_names)
events = df.toPandas()

# COMMAND ----------

# MAGIC %md ### Country Select and Feature Casting
# MAGIC     

# COMMAND ----------

# Country select
country_selected = 'AR'  # Reemplaza 'CH' con el nombre del país que deseas filtrar
events_filtered = events[events.Country == country_selected].copy()

# DATE to index
events_filtered['DATE'] = pd.to_datetime(events_filtered['DATE'])
events_filtered.set_index('DATE', inplace=True)
events_filtered.sort_index(ascending=True, inplace=True)

# Cast object to float
events_filtered['GoldsteinScaleWA'] = events_filtered['GoldsteinScaleWA'].astype(float)
events_filtered['ToneWA'] = events_filtered['ToneWA'].astype(float)

# COMMAND ----------

events_filtered

# COMMAND ----------

# MAGIC %md ### Feature Engineering
# MAGIC

# COMMAND ----------

# Feature Engineering

events_filtered['day'] = events_filtered.index.day
events_filtered['week'] = events_filtered.index.week
events_filtered['month'] = events_filtered.index.month
events_filtered['year'] = events_filtered.index.year
events_filtered['day_of_week'] = events_filtered.index.dayofweek

# Cyclical features
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

# features lag
events_filtered['GoldsteinScaleWA_lag1'] = events_filtered['GoldsteinScaleWA'].shift(1)
events_filtered['GoldsteinScaleWA_lag7'] = events_filtered['GoldsteinScaleWA'].shift(7)
events_filtered['GoldsteinScaleWA_lag30'] = events_filtered['GoldsteinScaleWA'].shift(30)
events_filtered['ToneWA_lag1'] = events_filtered['ToneWA'].shift(1)

# features movil windows
events_filtered['GoldsteinScaleWA_roll7'] = events_filtered['GoldsteinScaleWA'].rolling(window=7).mean()
events_filtered['ToneWA_roll7'] = events_filtered['ToneWA'].rolling(window=7).mean()

events_filtered.dropna(inplace=True)

# COMMAND ----------

# MAGIC %md ### Split & Train with best params
# MAGIC

# COMMAND ----------

X = events_filtered[['day_sin', 'day_cos', 'week_sin', 'week_cos', 'month_sin', 'month_cos', 
                     'year_sin', 'year_cos', 'day_of_week_sin', 'day_of_week_cos', 
                     'GoldsteinScaleWA_lag1', 'GoldsteinScaleWA_lag7', 'GoldsteinScaleWA_lag30', 
                     'ToneWA_lag1', 'GoldsteinScaleWA_roll7', 'ToneWA_roll7']]

y = events_filtered['GoldsteinScaleWA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

best_rf = RandomForestRegressor(
    max_depth=4, 
    max_features=1.0, 
    min_samples_leaf=6,
    min_samples_split=30, 
    n_estimators=200
)

best_rf.fit(X_train, y_train)

# Hacer predicciones
predictions = best_rf.predict(X_test)


# COMMAND ----------

# MAGIC %md ### Predict & Plot
# MAGIC

# COMMAND ----------

# Realizar predicciones
y_pred = best_rf.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error con los mejores parámetros: {mse}")

rmse = mean_squared_error(y_test, y_pred, squared=False)  # squared=False para obtener la raíz cuadrada
print(f"Root Mean Squared Error (RMSE): {rmse}")
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score (R-squared): {r2}")

# Crear la gráfica combinada
plt.figure(figsize=(14, 7))

# Graficar los valores reales de GoldsteinScale en el conjunto de entrenamiento
plt.plot(y_train.index, y_train, label='Valor Real - Train Set', color='blue')

# Graficar las predicciones de GoldsteinScale en el conjunto de prueba
plt.plot(y_test.index, y_test_pred, label='Predicción - Test Set', color='red', linestyle='--')

# Graficar los valores reales de GoldsteinScale en el conjunto de prueba
plt.plot(y_test.index, y_test, label='Valor Real - Test Set', color='green')

# Configuraciones de la gráfica
plt.title('Tendencia de GoldsteinScale: Valores Reales (Train), Predicciones (Test) y Valores Reales (Test)')
plt.xlabel('Date')
plt.ylabel('GoldsteinScale')
plt.ylim([-5, 5])
plt.xlim([pd.Timestamp('2024-06-01'), pd.Timestamp('2024-08-10')]) 
plt.legend()
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %pip install sqlalchemy

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sqlalchemy import create_engine
df_combined = pd.DataFrame()

# 1. Función para la conexión con Azure y Spark (ajustada)
def load_data_from_azure(storage_account_name, container_name, file_name, spark):
    file_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/{file_name}"
    df = spark.read.format("csv").option("header", "false").load(file_path)
    
    # Mostrar las primeras filas para ver los nombres de las columnas
    df.show(5)
    return df

# 2. Función para preparar y filtrar los eventos por país (ajustada)
def filter_and_prepare_events(events, country_selected):
    # Renombrar columnas basadas en la inspección visual del DataFrame
    column_names = ["DATE", "Country", "GoldsteinScaleWA", "ToneWA"]  # Asegúrate de que estos nombres coincidan con los datos
    events = events.toDF(*column_names).toPandas()

    if 'Country' not in events.columns:
        print("Error: La columna 'Country' no se encontró en el DataFrame después de renombrar las columnas.")
        return None

    # Filtrar por país
    events_filtered = events[events.Country == country_selected].copy()

    # Convierte DATE a datetime y establece como índice
    events_filtered['DATE'] = pd.to_datetime(events_filtered['DATE'])
    events_filtered.set_index('DATE', inplace=True)
    events_filtered.sort_index(ascending=True, inplace=True)

    # Convierte columnas a float
    events_filtered['GoldsteinScaleWA'] = events_filtered['GoldsteinScaleWA'].astype(float)
    events_filtered['ToneWA'] = events_filtered['ToneWA'].astype(float)
    
    return events_filtered

# 3. Función para la ingeniería de características
def feature_engineering(events_filtered):
    # Agregar nuevas características
    events_filtered['day'] = events_filtered.index.day
    events_filtered['week'] = events_filtered.index.isocalendar().week
    events_filtered['month'] = events_filtered.index.month
    events_filtered['year'] = events_filtered.index.year
    events_filtered['day_of_week'] = events_filtered.index.dayofweek

    # Features cíclicas
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

# 4. Función para entrenar y evaluar el modelo (ajustada para manejar casos con pocas muestras)
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

# 5. Función para guardar los resultados en SQL (ajustada para incluir y_train y y_test)
def save_results_to_sql(train_dates, test_dates, y_pred_train, y_pred_test, y_train, y_test, country, output_file):
    # Crear DataFrame para train
    results_train = pd.DataFrame({
        'fecha': train_dates,
        'pais': country,
        'y_pred': y_pred_train,
        'y_real': y_train
    })

    # Crear DataFrame para test
    results_test = pd.DataFrame({
        'fecha': test_dates,
        'pais': country,
        'y_pred': y_pred_test,
        'y_real': y_test
    })

    # Concatenar ambos resultados
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
        "password": dbutils.secrets.get(scope="events", key="ASQLPassword"),
        "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
    }

    # Write the DataFrame to SQL Server
    spark_result = spark.createDataFrame(df_combined)

    # Define the target table name
    table_name = "events.goldsteinPredictionsGold"

    # Write the Spark DataFrame to Azure SQL Database
    spark_result.write \
        .jdbc(url=jdbc_url, table=table_name, mode='overwrite', properties=connection_properties)


# 6. Función principal
def main():
    storage_account_name = "factoredatathon2024"
    container_name = "gold"
    file_name = "weightedAvgGoldsteinToneGold.csv"
    # output_file = "model_predictions.csv"

    storage_account_key = dbutils.secrets.get(scope="events", key="DataLakeKey")
    storage_account_name = "factoredatathon2024"
    container_name = "gold"

    spark.conf.set(
        f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net",
        f"{storage_account_key}"
    )

    file_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/weightedAvgGoldsteinToneGold.csv"
    df = spark.read.format("csv").option("header", "false").load(file_path)

    # Conexión y carga de datos
    df = load_data_from_azure(storage_account_name, container_name, file_name, spark)
    
    # Lista de países
    countries = df.select('_c1').distinct().rdd.flatMap(lambda x: x).collect()


    for country in countries:
        print(f"Processing country: {country}")
        # Preparación y filtrado de eventos
        events_filtered = filter_and_prepare_events(df, country)
        
        if events_filtered is not None and len(events_filtered) > 0:
            # Ingeniería de características
            events_filtered = feature_engineering(events_filtered)
            
            # Entrenamiento del modelo y obtención de predicciones
            train_dates, test_dates, y_pred_train, y_pred_test, y_train, y_test = train_model_and_get_predictions(events_filtered)
            
            if train_dates is not None:
                # Guardar resultados en SQL
                save_results_to_sql(train_dates, test_dates, y_pred_train, y_pred_test, y_train, y_test, country)


# Ejecutar el script
if __name__ == "__main__":
    main()

# COMMAND ----------


