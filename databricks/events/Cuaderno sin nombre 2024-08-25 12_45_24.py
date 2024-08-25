# Databricks notebook source
# MAGIC %md
# MAGIC Library Imports and Database Connection:

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

storage_account_name = "factoredatathon2024"
container_name = "gold"
file_name = "weightedAvgGoldsteinToneGold.csv"
output_file = "model_predictions.csv"

storage_account_key = dbutils.secrets.get(scope="events", key="DataLakeKey")
spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net",
    f"{storage_account_key}"
)

file_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/{file_name}"
df = spark.read.format("csv").option("header", "false").load(file_path)

df.show(5)

# COMMAND ----------

column_names = ["DATE", "Country", "GoldsteinScaleWA", "ToneWA"]
events =df
events = events.toDF(*column_names).toPandas()

country_selected = "US"

events_filtered = events[events.Country == country_selected].copy()

events_filtered['DATE'] = pd.to_datetime(events_filtered['DATE'])
events_filtered.set_index('DATE', inplace=True)
events_filtered.sort_index(ascending=True, inplace=True)

events_filtered['GoldsteinScaleWA'] = events_filtered['GoldsteinScaleWA'].astype(float)
events_filtered['ToneWA'] = events_filtered['ToneWA'].astype(float)

events_filtered.head()

# COMMAND ----------

events_filtered.tail()

# COMMAND ----------

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
events_filtered

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
y_pred_train = best_rf.predict(X_train)
y_pred_test = best_rf.predict(X_test)

train_dates, test_dates, y_pred_train, y_pred_test, y_train, y_test, model = X_train.index, X_test.index, y_pred_train, y_pred_test, y_train, y_test, best_rf

# COMMAND ----------

country="US"
df_combined = pd.DataFrame()

results_train = pd.DataFrame({
    'DATE': train_dates,
    'pais': country,
    'y_pred': y_pred_train,
    'y_real': y_train
})

results_test = pd.DataFrame({
    'DATE': test_dates,
    'pais': country,
    'y_pred': y_pred_test,
    'y_real': y_test
})

results = pd.concat([results_train, results_test])

global df_combined

df_combined = pd.concat([df_combined, results])
df_combined

# COMMAND ----------

from datetime import timedelta
import numpy as np
import pandas as pd

# Generar las fechas futuras
last_date = events_filtered.index.max()
future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]

# Crear un DataFrame para las fechas futuras
future_df = pd.DataFrame(index=future_dates)
future_df['day'] = future_df.index.day
future_df['week'] = future_df.index.isocalendar().week
future_df['month'] = future_df.index.month
future_df['year'] = future_df.index.year
future_df['day_of_week'] = future_df.index.dayofweek

# Crear las características de las fechas futuras
future_df['day_sin'] = np.sin(2 * np.pi * future_df['day'] / 31)
future_df['day_cos'] = np.cos(2 * np.pi * future_df['day'] / 31)
future_df['week_sin'] = np.sin(2 * np.pi * future_df['week'] / 52)
future_df['week_cos'] = np.cos(2 * np.pi * future_df['week'] / 52)
future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)
cycle_length = 10
future_df['year_sin'] = np.sin(2 * np.pi * future_df['year'] / cycle_length)
future_df['year_cos'] = np.cos(2 * np.pi * future_df['year'] / cycle_length)
future_df['day_of_week_sin'] = np.sin(2 * np.pi * future_df['day_of_week'] / 7)
future_df['day_of_week_cos'] = np.cos(2 * np.pi * future_df['day_of_week'] / 7)

# Asegúrate de que las características de lag y rolling estén alineadas con las fechas futuras
# Rellenar los NaN con los últimos valores disponibles (puedes ajustar esto según tus necesidades)
future_df['GoldsteinScaleWA_lag1'] = events_filtered['GoldsteinScaleWA'].shift(1).fillna(method='bfill').values[-len(future_dates):]
future_df['GoldsteinScaleWA_lag7'] = events_filtered['GoldsteinScaleWA'].shift(7).fillna(method='bfill').values[-len(future_dates):]
future_df['GoldsteinScaleWA_lag30'] = events_filtered['GoldsteinScaleWA'].shift(30).fillna(method='bfill').values[-len(future_dates):]
future_df['ToneWA_lag1'] = events_filtered['ToneWA'].shift(1).fillna(method='bfill').values[-len(future_dates):]

future_df['GoldsteinScaleWA_roll7'] = events_filtered['GoldsteinScaleWA'].rolling(window=7).mean().fillna(method='bfill').values[-len(future_dates):]
future_df['ToneWA_roll7'] = events_filtered['ToneWA'].rolling(window=7).mean().fillna(method='bfill').values[-len(future_dates):]

# Hacer las predicciones
X_future = future_df[['day_sin', 'day_cos', 'week_sin', 'week_cos', 'month_sin', 'month_cos', 
                      'year_sin', 'year_cos', 'day_of_week_sin', 'day_of_week_cos', 
                      'GoldsteinScaleWA_lag1', 'GoldsteinScaleWA_lag7', 'GoldsteinScaleWA_lag30',
                      'ToneWA_lag1', 'GoldsteinScaleWA_roll7', 'ToneWA_roll7']]

future_df['y_pred'] = best_rf.predict(X_future)
future_df['y_real'] = np.nan  # No hay valores reales aún

# Establecer el valor de la columna 'pais' para las fechas futuras
future_df['pais'] = country

# Preparar el DataFrame para concatenar
future_df.reset_index(inplace=True)
future_df.rename(columns={'index': 'DATE'}, inplace=True)

# Concatenar las predicciones al DataFrame df_combined
df_combined = pd.concat([df_combined, future_df[['DATE', 'pais', 'y_pred', 'y_real']]])




# COMMAND ----------

future_df

# COMMAND ----------

df_combined

# COMMAND ----------

import plotly.graph_objs as go
import plotly.express as px

# Asegúrate de que df_combined esté ordenado por fecha
df_combined.sort_values(by='DATE', inplace=True)

# Crear el gráfico
fig = go.Figure()

# Agregar las trazas para predicciones y valores reales
fig.add_trace(go.Scatter(x=df_combined['DATE'], y=df_combined['y_pred'], mode='lines', name='Predicción', line=dict(color='blue', dash='dash')))
fig.add_trace(go.Scatter(x=df_combined['DATE'], y=df_combined['y_real'], mode='lines', name='Valor Real', line=dict(color='red')))

# Configurar el diseño del gráfico
fig.update_layout(
    title='Predicción vs Valor Real',
    xaxis_title='Fecha',
    yaxis_title='Valor',
    legend_title='Leyenda',
    xaxis=dict(tickformat='%Y-%m-%d'),
    template='plotly_white'
)

# Mostrar el gráfico
fig.show()

