# Databricks notebook source
# MAGIC %pip install sqlalchemy

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pyodbc
import pandas as pd

jdbcHostname = "factoredata2024.database.windows.net"
jdbcPort = 1433
jdbcDatabase = "factoredata2024"

# Configura las credenciales
jdbcUsername = "factoredata2024admin"
jdbcPassword = dbutils.secrets.get(scope="events", key="ASQLPassword"),

# Configura la URL JDBC
jdbcUrl = f"jdbc:sqlserver://{jdbcHostname}:{jdbcPort};database={jdbcDatabase}"

# Configura las propiedades de conexión
connectionProperties = {
    "user" : jdbcUsername,
    "password" : jdbcPassword,
    "driver" : "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

# Realiza la consulta
query = "(SELECT fecha, y_pred, y_real FROM [events].[goldsteinPredictionsGold]) AS goldstein_data"

# Lee los datos en un DataFrame de Spark
df = spark.read.jdbc(url=jdbcUrl, table=query, properties=connectionProperties)

# Convierte a un DataFrame de Pandas si es necesario
pandas_df = df.toPandas()

# COMMAND ----------

import plotly.graph_objs as go

# Crea el gráfico de líneas
fig = go.Figure()

# Agrega la serie de y_pred
fig.add_trace(go.Scatter(x=pandas_df['fecha'], y=pandas_df['y_pred'], mode='lines', name='y_pred'))

# Agrega la serie de y_real
fig.add_trace(go.Scatter(x=pandas_df['fecha'], y=pandas_df['y_real'], mode='lines', name='y_real'))

# Configura los títulos y etiquetas
fig.update_layout(
    title="Predicciones vs Valores Reales",
    xaxis_title="Fecha",
    yaxis_title="Valores",
    legend_title="Series"
)

# Muestra el gráfico
fig.show()

# COMMAND ----------

import pandas as pd
from pyspark.sql import SparkSession

# Configura la sesión de Spark
spark = SparkSession.builder.appName("ReadData").getOrCreate()

# Parámetros de conexión JDBC
jdbcHostname = "factoredata2024.database.windows.net"
jdbcPort = 1433
jdbcDatabase = "dactoredata2024"
jdbcUsername = "factoredata2024admin"
jdbcPassword = dbutils.secrets.get(scope="events", key="ASQLPassword")

# Configuración de la URL JDBC
jdbcUrl = f"jdbc:sqlserver://{jdbcHostname}:{jdbcPort};database={jdbcDatabase}"

# Propiedades de conexión
connectionProperties = {
    "user": jdbcUsername,
    "password": jdbcPassword,
    "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

# Consulta SQL
query = "(SELECT fecha, pais, y_pred, y_real FROM [events].[goldsteinPredictionsGold]) AS goldstein_data"

# Leer los datos en un DataFrame de Spark
df_spark = spark.read.jdbc(url=jdbcUrl, table=query, properties=connectionProperties)

# Convertir el DataFrame de Spark a Pandas si es necesario
df_pandas = df_spark.toPandas()

# Crear el gráfico con Plotly



# COMMAND ----------

import plotly.express as px
df_pandas['fecha'] = pd.to_datetime(df_pandas['fecha'])
df_pandas.sort_values(by='fecha', inplace=True)

# Seleccionar el país manualmente
pais_seleccionado = 'AR'  # Cambia este valor al país que deseas visualizar

# Filtrar los datos por el país seleccionado
df_filtrado = df_pandas[df_pandas['pais'] == pais_seleccionado]

# Crear el gráfico con Plotly
import plotly.express as px
fig = px.line(df_filtrado, x='fecha', y=['y_pred', 'y_real'], title=f'Predicciones vs Valores Reales para {pais_seleccionado}')
fig.show()




# COMMAND ----------

!pip install plotly

# COMMAND ----------

fips_to_country = {
    'AF': 'Afghanistan',
    'AX': 'Akrotiri',
    'AL': 'Albania',
    'AG': 'Algeria',
    'AQ': 'American Samoa',
    'AN': 'Andorra',
    'AO': 'Angola',
    'AV': 'Anguilla',
    'AY': 'Antarctica',
    'AC': 'Antigua and Barbuda',
    'AR': 'Argentina',
    'AM': 'Armenia',
    'AA': 'Aruba',
    'AT': 'Ashmore and Cartier Islands',
    'AS': 'Australia',
    'AU': 'Austria',
    'AJ': 'Azerbaijan',
    'BF': 'Bahamas, The',
    'BA': 'Bahrain',
    'FQ': 'Baker Island',
    'BG': 'Bangladesh',
    'BB': 'Barbados',
    'BS': 'Bassas da India',
    'BO': 'Belarus',
    'BE': 'Belgium',
    'BH': 'Belize',
    'BN': 'Benin',
    'BD': 'Bermuda',
    'BT': 'Bhutan',
    'BL': 'Bolivia',
    'BK': 'Bosnia and Herzegovina',
    'BC': 'Botswana',
    'BV': 'Bouvet Island',
    'BR': 'Brazil',
    'IO': 'British Indian Ocean Territory',
    'BX': 'Brunei',
    'BU': 'Bulgaria',
    'UV': 'Burkina Faso',
    'BM': 'Burma',
    'BY': 'Burundi',
    'CB': 'Cambodia',
    'CM': 'Cameroon',
    'CA': 'Canada',
    'CV': 'Cape Verde',
    'CJ': 'Cayman Islands',
    'CT': 'Central African Republic',
    'CD': 'Chad',
    'CI': 'Chile',
    'CH': 'China',
    'KT': 'Christmas Island',
    'IP': 'Clipperton Island',
    'CK': 'Cocos (Keeling) Islands',
    'CO': 'Colombia',
    'CN': 'Comoros',
    'CG': 'Congo, Democratic Republic of the',
    'CF': 'Congo, Republic of the',
    'CW': 'Cook Islands',
    'CR': 'Coral Sea Islands',
    'CS': 'Costa Rica',
    'IV': "Cote d'Ivoire",
    'HR': 'Croatia',
    'CU': 'Cuba',
    'UC': 'Curacao',
    'CY': 'Cyprus',
    'EZ': 'Czech Republic',
    'DA': 'Denmark',
    'DX': 'Dhekelia',
    'DJ': 'Djibouti',
    'DO': 'Dominica',
    'DR': 'Dominican Republic',
    'EC': 'Ecuador',
    'EG': 'Egypt',
    'ES': 'El Salvador',
    'EK': 'Equatorial Guinea',
    'ER': 'Eritrea',
    'EN': 'Estonia',
    'ET': 'Ethiopia',
    'PJ': 'Etorofu, Habomai, Kunashiri, and Shikotan Islands',
    'EU': 'Europa Island',
    'FK': 'Falkland Islands (Islas Malvinas)',
    'FO': 'Faroe Islands',
    'FJ': 'Fiji',
    'FI': 'Finland',
    'FR': 'France',
    'FG': 'French Guiana',
    'FP': 'French Polynesia',
    'FS': 'French Southern and Antarctic Lands',
    'GB': 'Gabon',
    'GA': 'Gambia, The',
    'GZ': 'Gaza Strip',
    'GG': 'Georgia',
    'GM': 'Germany',
    'GH': 'Ghana',
    'GI': 'Gibraltar',
    'GO': 'Glorioso Islands',
    'GR': 'Greece',
    'GL': 'Greenland',
    'GJ': 'Grenada',
    'GP': 'Guadeloupe',
    'GQ': 'Guam',
    'GT': 'Guatemala',
    'GK': 'Guernsey',
    'GV': 'Guinea',
    'PU': 'Guinea-Bissau',
    'GY': 'Guyana',
    'HA': 'Haiti',
    'HM': 'Heard Island and McDonald Islands',
    'HO': 'Honduras',
    'HK': 'Hong Kong',
    'HQ': 'Howland Island',
    'HU': 'Hungary',
    'IC': 'Iceland',
    'IN': 'India',
    'ID': 'Indonesia',
    'IR': 'Iran',
    'IZ': 'Iraq',
    'EI': 'Ireland',
    'IM': 'Isle of Man',
    'IS': 'Israel',
    'IT': 'Italy',
    'JM': 'Jamaica',
    'JN': 'Jan Mayen',
    'JA': 'Japan',
    'DQ': 'Jarvis Island',
    'JE': 'Jersey',
    'JQ': 'Johnston Atoll',
    'JO': 'Jordan',
    'JU': 'Juan de Nova Island',
    'KZ': 'Kazakhstan',
    'KE': 'Kenya',
    'KQ': 'Kingman Reef',
    'KR': 'Kiribati',
    'KN': 'Korea, North',
    'KS': 'Korea, South',
    'KV': 'Kosovo',
    'KU': 'Kuwait',
    'KG': 'Kyrgyzstan',
    'LA': 'Laos',
    'LG': 'Latvia',
    'LE': 'Lebanon',
    'LT': 'Lesotho',
    'LI': 'Liberia',
    'LY': 'Libya',
    'LS': 'Liechtenstein',
    'LH': 'Lithuania',
    'LU': 'Luxembourg',
    'MC': 'Macau',
    'MK': 'Macedonia',
    'MA': 'Madagascar',
    'MI': 'Malawi',
    'MY': 'Malaysia',
    'MV': 'Maldives',
    'ML': 'Mali',
    'MT': 'Malta',
    'RM': 'Marshall Islands',
    'MB': 'Martinique',
    'MR': 'Mauritania',
    'MP': 'Mauritius',
    'MF': 'Mayotte',
    'MX': 'Mexico',
    'FM': 'Micronesia, Federated States of',
    'MQ': 'Midway Islands',
    'MD': 'Moldova',
    'MN': 'Monaco',
    'MG': 'Mongolia',
    'MJ': 'Montenegro',
    'MH': 'Montserrat',
    'MO': 'Morocco',
    'MZ': 'Mozambique',
    'BM': 'Myanmar',
    'WA': 'Namibia',
    'NR': 'Nauru',
    'BQ': 'Navassa Island',
    'NP': 'Nepal',
    'NL': 'Netherlands',
    'NC': 'New Caledonia',
    'NZ': 'New Zealand',
    'NU': 'Nicaragua',
    'NG': 'Niger',
    'NE': 'Niue',
    'NF': 'Norfolk Island',
    'CQ': 'Northern Mariana Islands',
    'NO': 'Norway',
    'MU': 'Oman',
    'PK': 'Pakistan',
    'PS': 'Palau',
    'LQ': 'Palmyra Atoll',
    'PM': 'Panama',
    'PP': 'Papua New Guinea',
    'PF': 'Paracel Islands',
    'PA': 'Paraguay',
    'PE': 'Peru',
    'RP': 'Philippines',
    'PC': 'Pitcairn Islands',
    'PL': 'Poland',
    'PO': 'Portugal',
    'RQ': 'Puerto Rico',
    'QA': 'Qatar',
    'RE': 'Reunion',
    'RO': 'Romania',
    'RS': 'Russia',
    'RW': 'Rwanda',
    'TB': 'Saint Barthelemy',
    'SH': 'Saint Helena',
    'SC': 'Saint Kitts and Nevis',
    'ST': 'Saint Lucia',
    'RN': 'Saint Martin',
    'SB': 'Saint Pierre and Miquelon',
    'VC': 'Saint Vincent and the Grenadines',
    'WS': 'Samoa',
    'SM': 'San Marino',
    'TP': 'Sao Tome and Principe',
    'SA': 'Saudi Arabia',
    'SG': 'Senegal',
    'RI': 'Serbia',
    'SE': 'Seychelles',
    'SL': 'Sierra Leone',
    'SN': 'Singapore',
    'NN': 'Sint Maarten',
    'LO': 'Slovakia',
    'SI': 'Slovenia',
    'BP': 'Solomon Islands',
    'SO': 'Somalia',
    'SF': 'South Africa',
    'SX': 'South Georgia and the Islands',
    'OD': 'South Sudan',
    'SP': 'Spain',
    'PG': 'Spratly Islands',
    'CE': 'Sri Lanka',
    'SU': 'Sudan',
    'NS': 'Suriname',
    'SV': 'Svalbard',
    'WZ': 'Swaziland',
    'SW': 'Sweden',
    'SZ': 'Switzerland',
    'SY': 'Syria',
    'TW': 'Taiwan',
    'TI': 'Tajikistan',
    'TZ': 'Tanzania',
    'TH': 'Thailand',
    'TT': 'Timor-Leste',
    'TO': 'Togo',
    'TL': 'Tokelau',
    'TN': 'Tonga',
    'TD': 'Trinidad and Tobago',
    'TE': 'Tromelin Island',
    'TS': 'Tunisia',
    'TU': 'Turkey',
    'TX': 'Turkmenistan',
    'TK': 'Turks and Caicos Islands',
    'TV': 'Tuvalu',
    'UG': 'Uganda',
    'UP': 'Ukraine',
    'AE': 'United Arab Emirates',
    'UK': 'United Kingdom',
    'US': 'United States',
    'UM': 'United States Minor Outlying Islands',
    'UY': 'Uruguay',
    'UZ': 'Uzbekistan',
    'NH': 'Vanuatu',
    'VT': 'Vatican City',
    'VE': 'Venezuela',
    'VM': 'Vietnam',
    'VI': 'Virgin Islands, British',
    'VQ': 'Virgin Islands, U.S.',
    'WQ': 'Wake Island',
    'WF': 'Wallis and Futuna',
    'WE': 'West Bank',
    'WI': 'Western Sahara',
    'YM': 'Yemen',
    'ZA': 'Zambia',
    'ZI': 'Zimbabwe'
}

# COMMAND ----------


pais_seleccionado_fips = "US"

def get_plot(pais_seleccionado_fips):
    pais_seleccionado = fips_to_country.get(pais_seleccionado_fips, 'Desconocido')
    df_filtrado = df_pandas[df_pandas['pais'] == pais_seleccionado_fips]
    fig = px.line(df_filtrado, x='fecha', y=['y_pred', 'y_real'], title=f'Predicciones vs Valores Reales para {pais_seleccionado}')
    return fig

get_plot("US")

# COMMAND ----------

country_to_fips = {v: k for k, v in fips_to_country.items()}

def get_plot(pais_seleccionado):
    # Obtener el código FIPS del país ingresado
    pais_seleccionado_fips = country_to_fips.get(pais_seleccionado, 'Desconocido')
    
    if pais_seleccionado_fips == 'Desconocido':
        return f'País "{pais_seleccionado}" no reconocido.'

    # Filtrar el DataFrame por el código FIPS del país
    df_filtrado = df_pandas[df_pandas['pais'] == pais_seleccionado_fips]
    
    if df_filtrado.empty:
        return f'No hay datos disponibles para el país "{pais_seleccionado}".'
    
    # Crear el gráfico
    fig = px.line(df_filtrado, x='fecha', y=['y_pred', 'y_real'], title=f'Predicciones vs Valores Reales para {pais_seleccionado}')
    
    return fig

# Ejemplo de uso
pais_ingresado = input("Ingrese el nombre del país: ")
figura = get_plot(pais_ingresado)

# Mostrar la figura (esto depende del entorno en el que estés trabajando)
figura.show()

# COMMAND ----------

import pandas as pd
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display


# Diccionario de países a códigos FIPS (invertido del diccionario anterior)
country_to_fips = {v: k for k, v in fips_to_country.items()}

# Convertir los nombres de los países a minúsculas para la búsqueda insensible a mayúsculas
country_names = [name.lower() for name in country_to_fips.keys()]

# Crear un widget de lista desplegable
dropdown = widgets.Combobox(
    placeholder='Selecciona un país',
    options=country_names,
    ensure_option=True,
    description='País:',
)

# Función para actualizar el gráfico basado en la selección del usuario
def update_plot(change):
    pais_seleccionado = change.new
    pais_seleccionado_normalizado = pais_seleccionado.lower()
    
    # Obtener el nombre del país desde el valor del widget
    nombre_pais = next((k for k, v in country_to_fips.items() if v.lower() == pais_seleccionado_normalizado), None)
    
    if nombre_pais is None:
        fig = px.line()  # Vacío o gráfico de error
        fig.update_layout(title="País no encontrado")
    else:
        pais_seleccionado_fips = country_to_fips.get(nombre_pais, 'Desconocido')
        df_filtrado = df_pandas[df_pandas['pais'] == pais_seleccionado_fips]
        if df_filtrado.empty:
            fig = px.line()  # Vacío o gráfico de error
            fig.update_layout(title=f'No hay datos disponibles para {nombre_pais}')
        else:
            fig = px.line(df_filtrado, x='fecha', y=['y_pred', 'y_real'], title=f'Predicciones vs Valores Reales para {nombre_pais}')
    
    # Mostrar el gráfico
    fig.show()

# Conectar el widget con la función de actualización
dropdown.observe(update_plot, names='value')

# Mostrar el widget en el notebook
display(dropdown)


# COMMAND ----------


