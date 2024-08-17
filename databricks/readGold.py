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

# MAGIC %md
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

zoom_start_train = 1000
zoom_end_train = 40000

# Asegurarse de que el rango de índices esté dentro de los límites
zoom_start_train = max(zoom_start_train, 0)
zoom_end_train = min(zoom_end_train, len(y_train))

# Crear la gráfica combinada con zoom
plt.figure(figsize=(14, 7))

# Graficar los valores reales de GoldsteinScale en el conjunto de entrenamiento
plt.plot(y_train.index[zoom_start_train:zoom_end_train], y_train.iloc[zoom_start_train:zoom_end_train],
         label='Valor Real - Train Set', color='blue')

# Graficar las predicciones de GoldsteinScale en el conjunto de prueba
plt.plot(y_test.index, y_test_pred,
         label='Predicción - Test Set', color='red', linestyle='--')

# Configuraciones de la gráfica
plt.title('Tendencia de GoldsteinScale: Valores Reales (Train) y Predicciones (Test) con Zoom (Índices 30000-40000)')
plt.xlabel('Índice')
plt.ylabel('GoldsteinScale')
plt.legend()
plt.grid(True)
plt.show()

# COMMAND ----------


