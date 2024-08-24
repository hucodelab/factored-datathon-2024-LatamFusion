# Databricks notebook source
# Import required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

%pip install geopandas
import geopandas as gpd
import requests
import zipfile
import io
import plotly.express as px
from datetime import datetime

# COMMAND ----------

# Load predictions and real data DF
df = pd.read_csv("model_predictions.csv")
df["fecha"] = pd.to_datetime(df["fecha"])
df.head(3)

# COMMAND ----------


# Function to plot predictions vs real values
def plot_predictions_vs_real(df, pais, y_min=-5, y_max=5):

    # Sort DF by date
    df = df.sort_values(by=['fecha'])

    # Filter the DataFrame for the selected country
    df_pais = df[df['pais'] == pais]

    plt.figure(figsize=(10, 6))

    # Plot predictions and real values
    plt.plot(df_pais['fecha'], df_pais['y_real'], label='Real', color='blue')
    plt.plot(df_pais['fecha'], df_pais['y_pred'], label='Pred', color='red', linestyle='--')

    # Format the x-axis to show months
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

    # Rotate x-axis labels for better visibility
    plt.gcf().autofmt_xdate()

    plt.grid(True)

    # Add legend, title, and labels
    plt.legend(loc='upper left')
    plt.title(f'Comparison of Predictions vs Real Values for {pais}')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.ylim([y_min, y_max])

    plt.show()



# COMMAND ----------

# Display 50 countries with most news
df["pais"].value_counts().head(50).plot(kind="bar", figsize=(14, 7))

# COMMAND ----------


# Plotting values for USA
plot_predictions_vs_real(df, 'US', y_min=-5, y_max=5)


# COMMAND ----------

# Plotting values for Russia
plot_predictions_vs_real(df, 'RS', y_min=-5, y_max=5)

# COMMAND ----------

# Plotting script
df = df.sort_values(by=['fecha'])

# Select the country you want to plot
pais = 'BR'

# Filter the DataFrame for the selected country
df_pais = df[df['pais'] == pais]

plt.figure(figsize=(10, 6))

# Plot predictions and real values
plt.plot(df_pais['fecha'], df_pais['y_real'], label='real', color='blue')
plt.plot(df_pais['fecha'], df_pais['y_pred'], label='pred', color='red', linestyle='--')

plt.legend(loc='upper left')
plt.title(f'Comparison of Predictions vs Real Values for {pais}')
plt.xlabel('Date')
plt.ylabel('Values')

plt.show()


# COMMAND ----------

# Display predicted and real values
# Display Preds. mean
# Display halfway min-mean pred

# Ensure the DataFrame is sorted by date
df = df.sort_values(by=['fecha'])

# Select the country you want to plot
pais = 'BR'  # Replace with the name of the country you want to plot

# Filter the DataFrame for the selected country
df_pais = df[df['pais'] == pais]

# Set up the plot
plt.figure(figsize=(10, 6))

# Plot predictions and real values
plt.plot(df_pais['fecha'], df_pais['y_real'], label='Real', color='blue')
plt.plot(df_pais['fecha'], df_pais['y_pred'], label='Pred', color='red', linestyle='--')

# Calculate and plot the mean of predictions
y_pred_mean = df_pais['y_pred'].mean()
plt.axhline(y=y_pred_mean, color='green', linestyle='-', label='Mean Pred')

# Calculate the minimum value of predictions
y_pred_min = df_pais['y_pred'].min()

# Calculate the halfway point between the mean and the minimum value of predictions
y_halfway = (y_pred_mean + y_pred_min) * 0.5

plt.axhline(y=y_halfway, color='purple', linestyle='--', label='Halfway Mean-Min Pred')

plt.legend(loc='upper left')
plt.title(f'Comparison of Predictions vs Real Values for {pais}')
plt.xlabel('Date')
plt.ylabel('Values')

plt.grid(True)

plt.show()


# COMMAND ----------

# Function to generate alerts

def generate_alerts(df, pais):
    """
    Generates alerts when y_pred is lower than y_halfway in the future,
    with one alert per day, even if there are multiple records on the same day.

    Parameters:
    - df: DataFrame with columns ['fecha', 'pais', 'y_pred', 'y_real', 'y_pred_plus_one'].
    - pais: Name of the country for which to perform the analysis.
    """
    
    # Ensure the DataFrame is sorted by date
    df = df.sort_values(by=['fecha'])

    # Filter the DataFrame for the selected country
    df_pais = df[df['pais'] == pais]

    # Calculate threshold
    y_pred_mean = df_pais['y_pred'].mean()
    y_pred_min = df_pais['y_pred'].min()
    distance = y_pred_mean - y_pred_min
    threshold = y_pred_mean - 0.3 * distance  # Threshold at 30% distance from mean towards minimum

    # Filter future dates
    # today = pd.Timestamp.today()
    today = pd.Timestamp('2024-06-01')
    df_future = df_pais[df_pais['fecha'] > today]

    # Identify alerts
    alerts = df_future[df_future['y_pred'] < threshold]

    # Group by date and take the first record of each day
    alerts = alerts.groupby('fecha').first().reset_index()

    # Print or save alerts
    if not alerts.empty:
        print(f"Alerts for {pais}:")
        print(alerts[['fecha', 'y_pred', 'y_real']])
    else:
        print(f"No alerts for {pais}.")

# Assuming you have a DataFrame called 'df' with the necessary columns
generate_alerts(df, 'BR')


# COMMAND ----------

# Generate alerts for Brazil
generate_alerts(df, "BR")

# COMMAND ----------

# Dict. fips to iso
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


# Plot the worldmap warnings
# Date to Datetime
df['fecha'] = pd.to_datetime(df['fecha'])
df['iso_country'] = df['pais'].map(fips_to_iso)

# Filter the DataFrame to include only today's data
df_today = df[df['fecha'].dt.strftime('%Y-%m-%d') == "2024-08-13"]

custom_color_scale = [
    [-5, 'red'],     # Lowest value (red)
    [0, 'yellow'],   # Intermediate color (yellow)
    [5, 'green']     # Highest value (green)
]

# Create the interactive map
fig = px.choropleth(df_today, 
                    locations='iso_country', 
                    locationmode='ISO-3',
                    color='y_real',
                    color_continuous_scale='blues',  # Changed to predefined colorscale
                    title='World Map with y_real Index by Country')

fig.show()


# COMMAND ----------


