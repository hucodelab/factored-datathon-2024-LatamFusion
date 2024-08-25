# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import timedelta

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
df = spark.read.format("csv").option("header", "True").load(file_path)
column_names = ["DATE", "Country", "GoldsteinScaleWA", "ToneWA"]
events = df
events = events.toDF(*column_names).toPandas()

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
# df_combined.to_csv(output_file, index=False)

# COMMAND ----------

df_combined.reset_index(inplace=True, drop=True)

# COMMAND ----------

combined_sdf = spark.createDataFrame(df_combined)
server = "factoredata2024.database.windows.net"
db = "dactoredata2024"
user = "factoredata2024admin"
password = dbutils.secrets.get(scope="events", key="ASQLPassword")

# JDBC connection properties
jdbc_url = f"jdbc:sqlserver://{server}:1433;database={db};user={user}@{db};password={password};encrypt=true;trustServerCertificate=false;hostNameInCertificate=*.database.windows.net;loginTimeout=30;"

connection_properties = {
    "user": f"{user}@{server}",
    "password": password,
    "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

# Table name in Azure SQL Database
table_name = "[events].[goldsteinPredictionsGold]"

# Write DataFrame to Azure SQL Database
combined_sdf.write.jdbc(url=jdbc_url, table=table_name, mode="overwrite", properties=connection_properties)
