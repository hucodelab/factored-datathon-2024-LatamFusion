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



# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 1. Función para la conexión con Azure y Spark
def load_data_from_azure(storage_account_name, container_name, file_name, spark):
    file_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/{file_name}"
    df = spark.read.format("csv").option("header", "false").load(file_path)
    return df

# 2. Función para preparar y filtrar los eventos por país
def filter_and_prepare_events(events, country_selected):
    column_names = ["DATE", "Country", "GoldsteinScaleWA", "ToneWA"]
    events = events.toDF(*column_names).toPandas()
    
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

# 4. Función para entrenar y evaluar el modelo
def train_and_evaluate_model(events_filtered):
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
    y_pred = best_rf.predict(X_test)

    # Evaluar el modelo
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R^2 Score: {r2}")

    return y_train, y_test, y_pred

# 5. Función para graficar los resultados
def plot_results(y_train, y_test, y_pred):
    plt.figure(figsize=(14, 7))

    # Graficar los valores reales de GoldsteinScale en el conjunto de entrenamiento
    plt.plot(y_train.index, y_train, label='Valor Real - Train Set', color='blue')

    # Graficar las predicciones de GoldsteinScale en el conjunto de prueba
    plt.plot(y_test.index, y_pred, label='Predicción - Test Set', color='red', linestyle='--')

    # Graficar los valores reales de GoldsteinScale en el conjunto de prueba
    plt.plot(y_test.index, y_test, label='Valor Real - Test Set', color='green')

    plt.title('Tendencia de GoldsteinScale: Valores Reales (Train), Predicciones (Test) y Valores Reales (Test)')
    plt.xlabel('Date')
    plt.ylabel('GoldsteinScale')
    plt.ylim([-5, 5])
    plt.xlim([pd.Timestamp('2024-06-01'), pd.Timestamp('2024-08-10')])
    plt.legend()
    plt.grid(True)
    plt.show()

# Ejecución del pipeline completo
def main():
    storage_account_name = "factoredatathon2024"
    container_name = "gold"
    file_name = "weightedAvgGoldsteinToneGold.csv"

    # Conexión y carga de datos
    df = load_data_from_azure(storage_account_name, container_name, file_name, spark)
    
    # Preparación y filtrado de eventos
    country_selected = 'US'
    events_filtered = filter_and_prepare_events(df, country_selected)
    
    # Ingeniería de características
    events_filtered = feature_engineering(events_filtered)
    
    # Entrenamiento y evaluación del modelo
    y_train, y_test, y_pred = train_and_evaluate_model(events_filtered)
    
    # Graficar los resultados
    plot_results(y_train, y_test, y_pred)

    print()



# COMMAND ----------

main()

# COMMAND ----------


