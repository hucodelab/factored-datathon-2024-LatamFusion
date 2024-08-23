# Databricks notebook source
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

df = pd.read_csv("model_predictions.csv")
df["fecha"] = pd.to_datetime(df["fecha"])
df.head(3)

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_predictions_vs_real(df, pais, y_min=-5, y_max=5):
    """
    Grafica las predicciones vs valores reales para un país específico y añade una cuadrícula.

    Parámetros:
    - df: DataFrame con las columnas ['fecha', 'pais', 'y_pred', 'y_real'].
    - pais: Nombre del país para el que se desea realizar el gráfico.
    - y_min: Límite inferior del eje y (opcional, por defecto -5).
    - y_max: Límite superior del eje y (opcional, por defecto 5).
    """

    # Asegúrate de tener el DataFrame ordenado por fecha
    df = df.sort_values(by=['fecha'])

    # Filtra el DataFrame para el país seleccionado
    df_pais = df[df['pais'] == pais]

    # Configuración del gráfico
    plt.figure(figsize=(10, 6))

    # Graficar las predicciones y los valores reales
    plt.plot(df_pais['fecha'], df_pais['y_real'], label='Real', color='blue')
    plt.plot(df_pais['fecha'], df_pais['y_pred'], label='Pred', color='red', linestyle='--')

    # Formatear el eje x para mostrar meses
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

    # Rotar las etiquetas del eje x para mejor visibilidad
    plt.gcf().autofmt_xdate()

    # Añadir cuadrícula
    plt.grid(True)

    # Añadir leyenda, título y etiquetas
    plt.legend(loc='upper left')
    plt.title(f'Comparación de Predicciones vs Valores Reales para {pais}')
    plt.xlabel('Fecha')
    plt.ylabel('Valores')
    plt.ylim([y_min, y_max])

    # Mostrar el gráfico
    plt.show()



# COMMAND ----------

def plot_predictions_vs_real_plus_one(df, pais, y_min=-5, y_max=5):
    """
    Grafica los valores reales vs predicciones ajustadas para un país específico y añade una cuadrícula.

    Parámetros:
    - df: DataFrame con las columnas ['fecha', 'pais', 'y_pred_plus_one', 'y_real'].
    - pais: Nombre del país para el que se desea realizar el gráfico.
    - y_min: Límite inferior del eje y (opcional, por defecto -5).
    - y_max: Límite superior del eje y (opcional, por defecto 5).
    """

    # Asegúrate de tener el DataFrame ordenado por fecha
    df = df.sort_values(by=['fecha'])

    # Filtra el DataFrame para el país seleccionado
    df_pais = df[df['pais'] == pais]

    # Configuración del gráfico
    plt.figure(figsize=(10, 6))

    # Graficar las predicciones ajustadas y los valores reales
    plt.plot(df_pais['fecha'], df_pais['y_real'], label='Real', color='blue')
    plt.plot(df_pais['fecha'], df_pais['y_pred_plus_one'], label='Pred Plus One', color='orange', linestyle='--')

    # Formatear el eje x para mostrar meses
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

    # Rotar las etiquetas del eje x para mejor visibilidad
    plt.gcf().autofmt_xdate()

    # Añadir cuadrícula
    plt.grid(True)

    # Añadir leyenda, título y etiquetas
    plt.legend(loc='upper left')
    plt.title(f'Comparación de Valores Reales vs Predicciones Ajustadas para {pais}')
    plt.xlabel('Fecha')
    plt.ylabel('Valores')
    plt.ylim([y_min, y_max])

    # Mostrar el gráfico
    plt.show()


# COMMAND ----------

df["pais"].value_counts().head(90).plot(kind="bar", figsize=(14, 7))

# COMMAND ----------

df.query("pais == 'US'").count()

# COMMAND ----------



# COMMAND ----------

plot_predictions_vs_real(df, 'US', y_min=-5, y_max=5)


# COMMAND ----------

plot_predictions_vs_real(df, 'AR', y_min=-8, y_max=8)

# COMMAND ----------

plot_predictions_vs_real(df, 'BR', y_min=-5, y_max=5)

# COMMAND ----------

plot_predictions_vs_real(df, 'CI', y_min=-6, y_max=6)

# COMMAND ----------

plot_predictions_vs_real(df, 'VE', y_min=-6, y_max=6)

# COMMAND ----------



# COMMAND ----------

plot_predictions_vs_real(df, 'RS', y_min=-5, y_max=5)

# COMMAND ----------

plot_predictions_vs_real(df, 'UP', y_min=-5, y_max=5)

# COMMAND ----------

print(df["y_pred"].median())
print(df["y_real"].median())

# COMMAND ----------




# COMMAND ----------

# MAGIC %md 
# MAGIC

# COMMAND ----------

df = df.sort_values(by=['fecha'])

# Selecciona el país que quieres graficar
pais = 'BR'  # Reemplaza con el nombre del país que deseas graficar

# Filtra el DataFrame para el país seleccionado
df_pais = df[df['pais'] == pais]

# Configuración del gráfico
plt.figure(figsize=(10, 6))

# Graficar las predicciones y los valores reales
plt.plot(df_pais['fecha'], df_pais['y_real'], label='real', color='blue')
plt.plot(df_pais['fecha'], df_pais['y_pred'], label='pred', color='red', linestyle='--')

# Añadir leyenda, título y etiquetas
plt.legend(loc='upper left')
plt.title(f'Comparación de Predicciones vs Valores Reales para {pais}')
plt.xlabel('Fecha')
plt.ylabel('Valores')

# Mostrar el gráfico
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

# Asegúrate de tener el DataFrame ordenado por fecha
df = df.sort_values(by=['fecha'])

# Selecciona el país que quieres graficar
pais = 'BR'  # Reemplaza con el nombre del país que deseas graficar

# Filtra el DataFrame para el país seleccionado
df_pais = df[df['pais'] == pais]

# Configuración del gráfico
plt.figure(figsize=(10, 6))

# Graficar las predicciones y los valores reales
plt.plot(df_pais['fecha'], df_pais['y_real'], label='Real', color='blue')
plt.plot(df_pais['fecha'], df_pais['y_pred'], label='Pred', color='red', linestyle='--')

# Calcular y graficar la media de las predicciones
y_pred_mean = df_pais['y_pred'].mean()
plt.axhline(y=y_pred_mean, color='green', linestyle='-', label='Mean Pred')

# Añadir leyenda, título y etiquetas
plt.legend(loc='upper left')
plt.title(f'Comparación de Predicciones vs Valores Reales para {pais}')
plt.xlabel('Fecha')
plt.ylabel('Valores')

# Añadir cuadrícula
plt.grid(True)

# Mostrar el gráfico
plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt

# Asegúrate de tener el DataFrame ordenado por fecha
df = df.sort_values(by=['fecha'])

# Selecciona el país que quieres graficar
pais = 'BR'  # Reemplaza con el nombre del país que deseas graficar

# Filtra el DataFrame para el país seleccionado
df_pais = df[df['pais'] == pais]

# Configuración del gráfico
plt.figure(figsize=(10, 6))

# Graficar las predicciones y los valores reales
plt.plot(df_pais['fecha'], df_pais['y_real'], label='Real', color='blue')
plt.plot(df_pais['fecha'], df_pais['y_pred'], label='Pred', color='red', linestyle='--')

# Calcular y graficar la media de las predicciones
y_pred_mean = df_pais['y_pred'].mean()
plt.axhline(y=y_pred_mean, color='green', linestyle='-', label='Mean Pred')

# Calcular el valor mínimo de las predicciones
y_pred_min = df_pais['y_pred'].min()

# Calcular la distancia media entre la media de las predicciones y el valor mínimo
y_halfway = (y_pred_mean + y_pred_min) * 0.5

# Graficar la línea horizontal adicional
plt.axhline(y=y_halfway, color='purple', linestyle='--', label='Halfway Mean-Min Pred')

# Añadir leyenda, título y etiquetas
plt.legend(loc='upper left')
plt.title(f'Comparación de Predicciones vs Valores Reales para {pais}')
plt.xlabel('Fecha')
plt.ylabel('Valores')

# Añadir cuadrícula
plt.grid(True)

# Mostrar el gráfico
plt.show()


# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

def generate_alerts(df, pais):
    """
    Genera alertas cuando y_pred sea menor que y_halfway en el futuro,
    con una alerta por día, aunque haya múltiples registros en un mismo día.

    Parámetros:
    - df: DataFrame con las columnas ['fecha', 'pais', 'y_pred', 'y_real', 'y_pred_plus_one'].
    - pais: Nombre del país para el que se desea realizar el análisis.
    """
    
    # Asegúrate de tener el DataFrame ordenado por fecha
    df = df.sort_values(by=['fecha'])

    # Filtra el DataFrame para el país seleccionado
    df_pais = df[df['pais'] == pais]

    # Calcular y_halfway
    #y_pred_mean = df_pais['y_pred'].mean()
    #y_pred_min = df_pais['y_pred'].min()
    #y_halfway = (y_pred_mean + y_pred_min) / 2
     
    # Calcular el umbral a una distancia del 30% desde la media hacia el mínimo
    y_pred_mean = df_pais['y_pred'].mean()
    y_pred_min = df_pais['y_pred'].min()
    distance = y_pred_mean - y_pred_min
    threshold = y_pred_mean - 0.3 * distance  # Umbral a 30% de distancia desde la media hacia el mínimo

    # Filtra las fechas futuras
    #today = pd.Timestamp.today()
    today = pd.Timestamp('2024-06-01')
    df_future = df_pais[df_pais['fecha'] > today]

    # Identificar alertas
    alerts = df_future[df_future['y_pred'] < threshold]

    # Agrupar por fecha y tomar el primer registro de cada día
    alerts = alerts.groupby('fecha').first().reset_index()

    # Imprimir o guardar las alertas
    if not alerts.empty:
        print(f"Alertas para {pais}:")
        print(alerts[['fecha', 'y_pred', 'y_real']])
    else:
        print(f"No hay alertas para {pais}.")

# Supongamos que tienes un DataFrame llamado 'df' con las columnas necesarias
generate_alerts(df, 'BR')



# COMMAND ----------

generate_alerts(df, "BR")

# COMMAND ----------

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


# Cargar un mapa del mundo con códigos FIPS de dos caracteres
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Convertir el código ISO de tres caracteres a dos caracteres usando una función de correspondencia
iso_to_fips = {
    'USA': 'US', 'CHN': 'CN', 'RUS': 'RU', # Otros países...
}
world['fips'] = world['iso_a2']  # Usando 'iso_a2' ya que es el código de dos caracteres

# Unir los datos agregados de GoldsteinScale con el mapa
world = world.merge(df_grouped, on='fips', how='left')

# Crear el mapa de calor basado en la escala de Goldstein
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
world.boundary.plot(ax=ax, linewidth=1)
world.plot(column='GoldsteinScale', cmap='coolwarm', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)

# Agregar título
plt.title('Mapa de Calor: Escala de Goldstein por País', fontsize=15)
plt.show()

