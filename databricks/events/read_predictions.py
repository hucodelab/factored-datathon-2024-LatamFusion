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

# MAGIC %pip install geopandas

# COMMAND ----------

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import requests
import zipfile
import io

# Descargar el archivo shapefile ZIP desde el repositorio de geopandas
url = 'https://github.com/geopandas/geopandas-data/raw/main/naturalearth_lowres.zip'
response = requests.get(url)

# Descomprimir el archivo ZIP
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    # Extraer todos los archivos
    z.extractall('naturalearth_lowres')

# Ruta al shapefile descargado y extraído
shapefile_path = 'naturalearth_lowres/ne_110m_admin_0_countries.shp'

# Supongamos que tienes tu DataFrame llamado df
data = {
    'codigo_fips': ['LT'],
    'y_real': [-2.059574]
}

df = pd.DataFrame(data)

# Cargar el shapefile del mapa mundial
world = gpd.read_file(shapefile_path)

# Asegúrate de que el dataset de geopandas tenga una columna para el código de país
# 'iso_a2' es la columna de códigos ISO de dos letras en este dataset
# Puedes renombrar la columna si es necesario para coincidir con 'codigo_fips'
world = world.rename(columns={'iso_a2': 'codigo_fips'})  # Ajusta si el nombre es diferente

# Merge de los datos del DataFrame con el DataFrame del mapa
world = world.merge(df, on='codigo_fips', how='left')

# Crear el mapa
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
world.boundary.plot(ax=ax)
world.plot(column='y_real', ax=ax, legend=True,
           legend_kwds={'label': "Índice y_real",
                        'orientation': "horizontal"},
           cmap='coolwarm')

plt.title('Mapa del Mundo con Índice y_real por País')
plt.show()


# COMMAND ----------

df.head()

# COMMAND ----------

fips_to_iso = {
    'AF': 'AFG',
    'AX': '-',
    'AL': 'ALB',
    'AG': 'DZA',
    'AQ': 'ASM',
    'AN': 'AND',
    'AO': 'AGO',
    'AV': 'AIA',
    'AY': 'ATA',
    'AC': 'ATG',
    'AR': 'ARG',
    'AM': 'ARM',
    'AA': 'ABW',
    'AT': 'AUS',
    'AS': 'AUS',
    'AU': 'AUT',
    'AJ': 'AZE',
    'BF': 'BHS',
    'BA': 'BHR',
    'FQ': 'UMI',
    'BG': 'BGD',
    'BB': 'BRB',
    'BS': 'REU',
    'BO': 'BLR',
    'BE': 'BEL',
    'BH': 'BLZ',
    'BN': 'BEN',
    'BD': 'BMU',
    'BT': 'BTN',
    'BL': 'BOL',
    'BK': 'BIH',
    'BC': 'BWA',
    'BV': 'BVT',
    'BR': 'BRA',
    'IO': 'IOT',
    'BX': 'BRN',
    'BU': 'BGR',
    'UV': 'BFA',
    'BM': 'MMR',
    'BY': 'BDI',
    'CB': 'KHM',
    'CM': 'CMR',
    'CA': 'CAN',
    'CV': 'CPV',
    'CJ': 'CYM',
    'CT': 'CAF',
    'CD': 'TCD',
    'CI': 'CHL',
    'CH': 'CHN',
    'KT': 'CXR',
    'IP': 'CLP',
    'CK': 'CCK',
    'CO': 'COL',
    'CN': 'COM',
    'CG': 'COD',
    'CF': 'COG',
    'CW': 'COK',
    'CR': 'AUS',
    'CS': 'CRI',
    'IV': 'CIV',
    'HR': 'HRV',
    'CU': 'CUB',
    'UC': 'CUW',
    'CY': 'CYP',
    'EZ': 'CZE',
    'DA': 'DNK',
    'DX': '-',
    'DJ': 'DJI',
    'DO': 'DMA',
    'DR': 'DOM',
    'EC': 'ECU',
    'EG': 'EGY',
    'ES': 'SLV',
    'EK': 'GNQ',
    'ER': 'ERI',
    'EN': 'EST',
    'ET': 'ETH',
    'PJ': '-',
    'EU': 'REU',
    'FK': 'FLK',
    'FO': 'FRO',
    'FJ': 'FJI',
    'FI': 'FIN',
    'FR': 'FRA',
    'FG': 'GUF',
    'FP': 'PYF',
    'FS': 'ATF',
    'GB': 'GAB',
    'GA': 'GMB',
    'GZ': 'PSE',
    'GG': 'GEO',
    'GM': 'DEU',
    'GH': 'GHA',
    'GI': 'GIB',
    'GO': 'REU',
    'GR': 'GRC',
    'GL': 'GRL',
    'GJ': 'GRD',
    'GP': 'GLP',
    'GQ': 'GUM',
    'GT': 'GTM',
    'GK': 'GBR',
    'GV': 'GIN',
    'PU': 'GNB',
    'GY': 'GUY',
    'HA': 'HTI',
    'HM': 'HMD',
    'HO': 'HND',
    'HK': 'HKG',
    'HQ': 'UMI',
    'HU': 'HUN',
    'IC': 'ISL',
    'IN': 'IND',
    'ID': 'IDN',
    'IR': 'IRN',
    'IZ': 'IRQ',
    'EI': 'IRL',
    'IM': 'GBR',
    'IS': 'ISR',
    'IT': 'ITA',
    'JM': 'JAM',
    'JN': 'SJM',
    'JA': 'JPN',
    'DQ': 'UMI',
    'JE': 'GBR',
    'JQ': 'UMI',
    'JO': 'JOR',
    'JU': 'REU',
    'KZ': 'KAZ',
    'KE': 'KEN',
    'KQ': 'UMI',
    'KR': 'KIR',
    'KN': 'PRK',
    'KS': 'KOR',
    'KV': '-',
    'KU': 'KWT',
    'KG': 'KGZ',
    'LA': 'LAO',
    'LG': 'LVA',
    'LE': 'LBN',
    'LT': 'LSO',
    'LI': 'LBR',
    'LY': 'LBY',
    'LS': 'LIE',
    'LH': 'LTU',
    'LU': 'LUX',
    'MC': 'MAC',
    'MK': 'MKD',
    'MA': 'MDG',
    'MI': 'MWI',
    'MY': 'MYS',
    'MV': 'MDV',
    'ML': 'MLI',
    'MT': 'MLT',
    'RM': 'MHL',
    'MB': 'MTQ',
    'MR': 'MRT',
    'MP': 'MUS',
    'MF': 'MYT',
    'MX': 'MEX',
    'FM': 'FSM',
    'MQ': 'UMI',
    'MD': 'MDA',
    'MN': 'MCO',
    'MG': 'MNG',
    'MJ': 'MNE',
    'MH': 'MSR',
    'MO': 'MAR',
    'MZ': 'MOZ',
    'BM': 'MMR',
    'WA': 'NAM',
    'NR': 'NRU',
    'BQ': 'UMI',
    'NP': 'NPL',
    'NL': 'NLD',
    'NC': 'NCL',
    'NZ': 'NZL',
    'NU': 'NIC',
    'NG': 'NER',
    'NI': 'NGA',
    'NE': 'NIU',
    'NF': 'NFK',
    'CQ': 'MNP',
    'NO': 'NOR',
    'MU': 'OMN',
    'PK': 'PAK',
    'PS': 'PLW',
    'LQ': 'UMI',
    'PM': 'PAN',
    'PP': 'PNG',
    'PF': '-',
    'PA': 'PRY',
    'PE': 'PER',
    'RP': 'PHL',
    'PC': 'PCN',
    'PL': 'POL',
    'PO': 'PRT',
    'RQ': 'PRI',
    'QA': 'QAT',
    'RE': 'REU',
    'RO': 'ROU',
    'RS': 'RUS',
    'RW': 'RWA',
    'TB': 'BLM',
    'SH': 'SHN',
    'SC': 'KNA',
    'ST': 'LCA',
    'RN': 'MTQ',
    'SB': 'SPM',
    'VC': 'VCT',
    'WS': 'WSM',
    'SM': 'SMR',
    'TP': 'STP',
    'SA': 'SAU',
    'SG': 'SEN',
    'RI': 'SRB',
    'SE': 'SYC',
    'SL': 'SLE',
    'SN': 'SGP',
    'NN': 'SXM',
    'LO': 'SVK',
    'SI': 'SVN',
    'BP': 'SLB',
    'SO': 'SOM',
    'SF': 'ZAF',
    'SX': 'SGS',
    'OD': 'SSD',
    'SP': 'ESP',
    'PG': '-',
    'CE': 'LKA',
    'SU': 'SDN',
    'NS': 'SUR',
    'SV': 'SJM',
    'WZ': 'SWZ',
    'SW': 'SWE',
    'SZ': 'CHE',
    'SY': 'SYR',
    'TW': 'TWN',
    'TI': 'TJK',
    'TZ': 'TZA',
    'TH': 'THA',
    'TT': 'TLS',
    'TO': 'TGO',
    'TL': 'TKL',
    'TN': 'TON',
    'TD': 'TTO',
    'TE': 'UMI',
    'TS': 'TUN',
    'TU': 'TUR',
    'TX': 'TKM',
    'TK': 'TCA',
    'TV': 'TUV',
    'UG': 'UGA',
    'UP': 'UKR',
    'AE': 'ARE',
    'UK': 'GBR',
    'US': 'USA',
    'UY': 'URY',
    'UZ': 'UZB',
    'NH': 'VUT',
    'VT': 'VAT',
    'VE': 'VEN',
    'VM': 'VNM',
    'VI': 'VGB',
    'VQ': 'VIR',
    '-': '-',
    '-': '-',
    'YM': 'YEM',
    '-': '-',
    'ZA': 'ZMB',
    'ZI': 'ZWE'
}


# COMMAND ----------

import pandas as pd
import plotly.express as px
from datetime import datetime

df['fecha'] = pd.to_datetime(df['fecha'])
df['iso_country'] = df['pais'].map(fips_to_iso)

# Filtrar el DataFrame para incluir solo los datos del día de hoy
df_hoy = df[df['fecha'].dt.strftime('%Y-%m-%d') == "2024-08-13"]

custom_color_scale = [
    [-5, 'red'],     # Valor más bajo (rojo)
    [0, 'yellow'], # Color intermedio (amarillo)
    [5, 'green']    # Valor más alto (verde)
]
# Crear el mapa interactivo
fig = px.choropleth(df_hoy, 
                    locations='iso_country', 
                    locationmode='ISO-3',
                    color='y_real',
                    color_continuous_scale='blues',  # Cambiado a colorescale predefinido
                    title='Mapa del Mundo con Índice y_real por País')

fig.show()


# COMMAND ----------


