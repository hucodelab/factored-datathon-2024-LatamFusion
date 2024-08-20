# Databricks notebook source
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

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

     

# COMMAND ----------

column_names = ["DATE", "Country", "GoldsteinScaleWA","ToneWA"]
df = df.toDF(*column_names)
events = df.toPandas()

display(events)

# COMMAND ----------

events.value_counts('Country')

# COMMAND ----------

# MAGIC %md ## Primer caso de estudio:
# MAGIC     

# COMMAND ----------

country_selected = 'CH'  # Reemplaza 'CH' con el nombre del país que deseas filtrar
events_filtered = events[events.Country == country_selected].copy()

# COMMAND ----------

events_filtered['DATE'] = pd.to_datetime(events_filtered['DATE'])
events_filtered.set_index('DATE', inplace=True)

# COMMAND ----------

#temp variables
#events_filtered.DATE = pd.DatetimeIndex(events_filtered.DATE)
#events_filtered['DATE'].dtype


# COMMAND ----------

events_filtered.head()

# COMMAND ----------

#events_filtered.index=pd.DatetimeIndex(events_filtered['DATE'], freq='D')
#events_filtered.drop(columns=['DATE'], inplace=True)

# COMMAND ----------

events_filtered.head()

# COMMAND ----------

events_filtered['GoldsteinScaleWA'] = events_filtered['GoldsteinScaleWA'].astype(float)

# COMMAND ----------

events_filtered['ToneWA'] = events_filtered['ToneWA'].astype(float)

# COMMAND ----------

events_filtered.dtypes

# COMMAND ----------

events_filtered['day'] = events_filtered.index.day
events_filtered['week'] = events_filtered.index.week
events_filtered['month'] = events_filtered.index.month
events_filtered['year'] = events_filtered.index.year
events_filtered['day_of_week'] = events_filtered.index.dayofweek

# COMMAND ----------

events_filtered.sort_index(ascending=True, inplace=True)
events_filtered.head()

# COMMAND ----------

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15, 10)
events_filtered.plot(kind = "line", y = ['GoldsteinScaleWA']);

# COMMAND ----------

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15, 10)
events_filtered.plot(kind = "line", y = ['ToneWA'], color='blue');

# COMMAND ----------

# features lag
events_filtered['GoldsteinScaleWA_lag1'] = events_filtered['GoldsteinScaleWA'].shift(1)
events_filtered['GoldsteinScaleWA_lag7'] = events_filtered['GoldsteinScaleWA'].shift(7)
events_filtered['GoldsteinScaleWA_lag30'] = events_filtered['GoldsteinScaleWA'].shift(30)
events_filtered['ToneWA_lag1'] = events_filtered['ToneWA'].shift(1)

# COMMAND ----------

# features movil windows
events_filtered['GoldsteinScaleWA_roll7'] = events_filtered['GoldsteinScaleWA'].rolling(window=7).mean()
events_filtered['ToneWA_roll7'] = events_filtered['ToneWA'].rolling(window=7).mean()

# COMMAND ----------

events_filtered.dropna(inplace=True)

# COMMAND ----------


X = events_filtered[['day', 'week', 'month', 'year', 'day_of_week', 'GoldsteinScaleWA_lag1','GoldsteinScaleWA_lag7','GoldsteinScaleWA_lag30', 'ToneWA_lag1','GoldsteinScaleWA_roll7', 'ToneWA_roll7']]
y = events_filtered['GoldsteinScaleWA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

time_model = RandomForestRegressor(n_estimators=100, random_state=42)
time_model.fit(X_train, y_train)

# COMMAND ----------

y_pred = time_model.predict(X_test)

# COMMAND ----------

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# COMMAND ----------

y_test_pred = time_model.predict(X_test)

# Crear la gráfica combinada
plt.figure(figsize=(14, 7))

# Graficar los valores reales de GoldsteinScale en el conjunto de entrenamiento
plt.plot(y_train.index, y_train, label='Valor Real - Train Set', color='blue')

# Graficar las predicciones de GoldsteinScale en el conjunto de prueba
plt.plot(y_test.index, y_test_pred, label='Predicción - Test Set', color='red', linestyle='--')

# Configuraciones de la gráfica
plt.title('Tendencia de GoldsteinScale: Valores Reales (Train) y Predicciones (Test)')
plt.xlabel('Date')
plt.ylabel('GoldsteinScale')
plt.ylim([-5, 5])
plt.legend()
plt.grid(True)
plt.show()

# COMMAND ----------

y_test_pred = time_model.predict(X_test)

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
plt.xlim([pd.Timestamp('2024-05-01'), pd.Timestamp('2024-08-10')]) 
plt.legend()
plt.grid(True)
plt.show()


# COMMAND ----------



# COMMAND ----------

# MAGIC %md ##Prueba con EEUU:

# COMMAND ----------

country_selected = 'US'  # Reemplaza 'CH' con el nombre del país que deseas filtrar
events_filtered = events[events.Country == country_selected].copy()

# COMMAND ----------

events_filtered['DATE'] = pd.to_datetime(events_filtered['DATE'])
events_filtered.set_index('DATE', inplace=True)

# COMMAND ----------

events_filtered['GoldsteinScaleWA'] = events_filtered['GoldsteinScaleWA'].astype(float)

# COMMAND ----------

events_filtered['ToneWA'] = events_filtered['ToneWA'].astype(float)

# COMMAND ----------

events_filtered['day'] = events_filtered.index.day
events_filtered['week'] = events_filtered.index.week
events_filtered['month'] = events_filtered.index.month
events_filtered['year'] = events_filtered.index.year
events_filtered['day_of_week'] = events_filtered.index.dayofweek

# COMMAND ----------

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


# COMMAND ----------

events_filtered.sort_index(ascending=True, inplace=True)
events_filtered.head()

# COMMAND ----------

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15, 10)
events_filtered.plot(kind = "line", y = ['GoldsteinScaleWA']);

# COMMAND ----------

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15, 10)
events_filtered.plot(kind = "line", y = ['ToneWA'], color='blue');

# COMMAND ----------

# features lag
events_filtered['GoldsteinScaleWA_lag1'] = events_filtered['GoldsteinScaleWA'].shift(1)
events_filtered['GoldsteinScaleWA_lag7'] = events_filtered['GoldsteinScaleWA'].shift(7)
events_filtered['GoldsteinScaleWA_lag30'] = events_filtered['GoldsteinScaleWA'].shift(30)
events_filtered['ToneWA_lag1'] = events_filtered['ToneWA'].shift(1)

# COMMAND ----------

# features lag
events_filtered['GoldsteinScaleWA_lag1'] = events_filtered['GoldsteinScaleWA'].shift(1)
events_filtered['GoldsteinScaleWA_lag7'] = events_filtered['GoldsteinScaleWA'].shift(7)
events_filtered['GoldsteinScaleWA_lag30'] = events_filtered['GoldsteinScaleWA'].shift(30)
events_filtered['ToneWA_lag1'] = events_filtered['ToneWA'].shift(1)

# COMMAND ----------

# features movil windows
events_filtered['GoldsteinScaleWA_roll7'] = events_filtered['GoldsteinScaleWA'].rolling(window=7).mean()
events_filtered['ToneWA_roll7'] = events_filtered['ToneWA'].rolling(window=7).mean()

# COMMAND ----------

events_filtered.dropna(inplace=True)

# COMMAND ----------


#X = events_filtered[['day', 'week', 'month', 'year', 'day_of_week', 'GoldsteinScaleWA_lag1','GoldsteinScaleWA_lag7','GoldsteinScaleWA_lag30', 'ToneWA_lag1','GoldsteinScaleWA_roll7', 'ToneWA_roll7']]

X = events_filtered[['day_sin', 'day_cos', 'week_sin', 'week_cos', 'month_sin', 'month_cos', 
                     'year_sin', 'year_cos', 'day_of_week_sin', 'day_of_week_cos', 
                     'GoldsteinScaleWA_lag1', 'GoldsteinScaleWA_lag7', 'GoldsteinScaleWA_lag30', 
                     'ToneWA_lag1', 'GoldsteinScaleWA_roll7', 'ToneWA_roll7']]

y = events_filtered['GoldsteinScaleWA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

time_model = RandomForestRegressor(n_estimators=100, random_state=42)
time_model.fit(X_train, y_train)

# COMMAND ----------

y_pred = time_model.predict(X_test)

# COMMAND ----------

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# COMMAND ----------

y_test_pred = time_model.predict(X_test)

# Crear la gráfica combinada
plt.figure(figsize=(14, 7))

# Graficar los valores reales de GoldsteinScale en el conjunto de entrenamiento
plt.plot(y_train.index, y_train, label='Valor Real - Train Set', color='blue')

# Graficar las predicciones de GoldsteinScale en el conjunto de prueba
plt.plot(y_test.index, y_test_pred, label='Predicción - Test Set', color='red', linestyle='--')

# Configuraciones de la gráfica
plt.title('Tendencia de GoldsteinScale: Valores Reales (Train) y Predicciones (Test)')
plt.xlabel('Date')
plt.ylabel('GoldsteinScale')
plt.ylim([-5, 5])
plt.legend()
plt.grid(True)
plt.show()

# COMMAND ----------

# <<<<<<< Updated upstream
# train = pd.DataFrame({
#     'y_train.index': y_train.index,
#     'y_train': y_train.values
# })

# test = pd.DataFrame({
#     'y_test.index': y_test.index,
#     'y_pred': y_test_pred
# })

# real = pd.DataFrame({
#     'y_test.index': y_test.index,
#     'y_real': y_test
# })
# y_test.index, y_test

# # Save DataFrame to a CSV file in Databricks
# csv_path = '/Workspace/Shared/real.csv'  # The /dbfs/ prefix is used to write files to DBFS

# real.to_csv(csv_path, index=False)

# # Verify that the file is saved correctly (optional)
# dbutils.fs.ls('dbfs:/Workspace/Shared/')
#=======
y_test_pred = time_model.predict(X_test)

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

# MAGIC %md ### Hyperparams Optimization:
# MAGIC

# COMMAND ----------

from sklearn.model_selection import GridSearchCV

# COMMAND ----------


#X = events_filtered[['day', 'week', 'month', 'year', 'day_of_week', 'GoldsteinScaleWA_lag1','GoldsteinScaleWA_lag7','GoldsteinScaleWA_lag30', 'ToneWA_lag1','GoldsteinScaleWA_roll7', 'ToneWA_roll7']]
X = events_filtered[['day_sin', 'day_cos', 'week_sin', 'week_cos', 'month_sin', 'month_cos', 
                     'year_sin', 'year_cos', 'day_of_week_sin', 'day_of_week_cos', 
                     'GoldsteinScaleWA_lag1', 'GoldsteinScaleWA_lag7', 'GoldsteinScaleWA_lag30', 
                     'ToneWA_lag1', 'GoldsteinScaleWA_roll7', 'ToneWA_roll7']]

y = events_filtered['GoldsteinScaleWA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Definir el modelo
rf = RandomForestRegressor()

# Definir el grid de hiperparámetros a probar
param_grid = {
    'n_estimators': [150,200,250,300,350],
    'max_depth': [None, 1, 2, 3, 4, 5, 10],
    'min_samples_split': [10, 20, 30],
    'min_samples_leaf': [4, 5, 6, 7],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Configurar GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Entrenar GridSearchCV
grid_search.fit(X_train, y_train)

# Mostrar los mejores parámetros encontrados
print("Mejores parámetros encontrados:")
print(grid_search.best_params_)

# Evaluar el modelo con los mejores parámetros
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error con los mejores parámetros: {mse}")

# COMMAND ----------

best_rf = grid_search.best_estimator_
best_rf

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

#Mejores parámetros encontrados:
#{'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 300}
#Mean Squared Error con los mejores parámetros: 0.12350352406916495

#Mejores parámetros encontrados:
#{'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 20, 'n_estimators': 200}
#Mean Squared Error con los mejores parámetros: 0.11963690587393314

#Mejores parámetros encontrados:
#{'max_depth': 3, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 20, 'n_estimators': 300}
#Mean Squared Error con los mejores parámetros: 0.11862474667284435

# COMMAND ----------


#X = events_filtered[['day', 'week', 'month', 'year', 'day_of_week', 'GoldsteinScaleWA_lag1','GoldsteinScaleWA_lag7','GoldsteinScaleWA_lag30', 'ToneWA_lag1','GoldsteinScaleWA_roll7', 'ToneWA_roll7']]
X = events_filtered[['day_sin', 'day_cos', 'week_sin', 'week_cos', 'month_sin', 'month_cos', 
                     'year_sin', 'year_cos', 'day_of_week_sin', 'day_of_week_cos', 
                     'GoldsteinScaleWA_lag1', 'GoldsteinScaleWA_lag7', 'GoldsteinScaleWA_lag30', 
                     'ToneWA_lag1', 'GoldsteinScaleWA_roll7', 'ToneWA_roll7']]

y = events_filtered['GoldsteinScaleWA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Definir el modelo
rf = RandomForestRegressor()

# Definir el grid de hiperparámetros a probar
param_grid = {
    'n_estimators': [150,200,250,300,350],
    'max_depth': [None, 1, 2, 3, 4, 5, 10],
    'min_samples_split': [10, 20, 30],
    'min_samples_leaf': [4, 5, 6, 7],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Configurar GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_absolute_error')

# Entrenar GridSearchCV
grid_search.fit(X_train, y_train)

# Mostrar los mejores parámetros encontrados
print("Mejores parámetros encontrados:")
print(grid_search.best_params_)

# Evaluar el modelo con los mejores parámetros
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error con los mejores parámetros: {mae}")

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

# MAGIC %md ##XGBoost + USA
# MAGIC

# COMMAND ----------

# Definir el modelo
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Definir el grid de hiperparámetros a probar
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3]
}

# Configurar GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_absolute_error')

# Entrenar GridSearchCV
grid_search.fit(X_train, y_train)

# Obtener los mejores parámetros y el mejor modelo
best_xgb = grid_search.best_estimator_

# Realizar predicciones
y_pred = best_xgb.predict(X_test)

# Calcular las métricas
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Mostrar las métricas
print(f"Best parameters: {grid_search.best_params_}")
print(f"Mean Squared Error con los mejores parámetros: {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R^2): {r2}")



# COMMAND ----------

# Mostrar las métricas
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error con los mejores parámetros: {mse}")

# COMMAND ----------

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

# MAGIC %md ## ARIMA + USA
# MAGIC

# COMMAND ----------


events_filtered['GoldsteinLog'] = np.log(events_filtered.GoldsteinScaleWA)
events_filtered["GoldsteinLogShift1"] = events_filtered.GoldsteinLog.shift()
events_filtered["GoldsteinLogDiff"] = events_filtered.GoldsteinLog - events_filtered.GoldsteinLogShift1
ts = events_filtered.GoldsteinLog
ts_diff = events_filtered.GoldsteinLogDiff
ts_diff.dropna(inplace = True)

# COMMAND ----------

from statsmodels.tsa.stattools import acf, pacf

# COMMAND ----------

lag_acf = acf(ts_diff, nlags=20)
lag_acf

# COMMAND ----------

ACF = pd.Series(lag_acf)
ACF.plot(kind = "bar")

# COMMAND ----------

lag_pacf = pacf(ts_diff, nlags=20, method='ols');

# COMMAND ----------

PACF = pd.Series(lag_pacf)
PACF.plot(kind = "bar");

# COMMAND ----------

# Veamos qué parámetros son significativamente distintos de cero

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

# COMMAND ----------

from statsmodels.tsa.arima.model import ARIMA
ts.head()

# COMMAND ----------

# Instancio el modelo con parámetros (p=1,d=0,q=1) según el análisis de ACF y PACF
# En este caso d=0 porque trabajamos directamente con las diferencias
model_AR1MA = ARIMA(ts_diff, order=(1,0,1))

# Fiteo el modelo
results_ARIMA = model_AR1MA.fit()
results_ARIMA.fittedvalues.head()

# COMMAND ----------

print(results_ARIMA.summary())

# COMMAND ----------

ts_diff.plot()
results_ARIMA.fittedvalues.plot();

# COMMAND ----------

ts_diff.sum()

# COMMAND ----------

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff.tail()


# COMMAND ----------

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_diff_cumsum.tail()

# COMMAND ----------

ts.iloc[0]

# COMMAND ----------

predictions_ARIMA_log = pd.Series(ts.iloc[0], index=ts.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.tail()

# COMMAND ----------

events_filtered['GoldsteinPredicted'] = np.exp(predictions_ARIMA_log)

# COMMAND ----------

rmse = mean_squared_error(events_filtered.GoldsteinPredicted, events_filtered.GoldsteinScaleWA, squared=False)
rmse


# COMMAND ----------


