# Databricks notebook source
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

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
     

# COMMAND ----------

column_names = ["DATE", "Country", "GoldsteinScaleWA","ToneWA"]
df = df.toDF(*column_names)
events = df.toPandas()

display(events)

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
plt.xlim([pd.Timestamp('2023-09-01'), pd.Timestamp('2024-08-10')]) 
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

<<<<<<< Updated upstream
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
=======
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
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# COMMAND ----------


X = events_filtered[['day', 'week', 'month', 'year', 'day_of_week', 'GoldsteinScaleWA_lag1','GoldsteinScaleWA_lag7','GoldsteinScaleWA_lag30', 'ToneWA_lag1','GoldsteinScaleWA_roll7', 'ToneWA_roll7']]
y = events_filtered['GoldsteinScaleWA']

X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Definir el modelo
rf = RandomForestRegressor()

# Definir el grid de hiperparámetros a probar
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
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

Mejores parámetros encontrados:
{'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
Mean Squared Error con los mejores parámetros: 9597.231581032345
>>>>>>> Stashed changes
