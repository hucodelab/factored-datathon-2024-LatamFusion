# import matplotlib.pyplot as plt
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pymssql import connect
from sqlalchemy import create_engine

### STREAMLIT CONFIGURATION ###

st.set_page_config(
    page_title="GDELT Dashboard",  # Title that appears on the browser tab
    page_icon="./images/LatamFusionOnlyLogo.png",  # Cambiar # Favicon, you can use an emoji or a custom image path
    layout="centered",  # Layout options: "centered" or "wide"
    initial_sidebar_state="auto",  # Sidebar state: "auto", "expanded", "collapsed"
)

### STREAMLIT CONFIGURATION ###


### DATA LOADING ###
# Azure SQL Database connection details
server = "factoredata2024.database.windows.net"
database = "dactoredata2024"
username = "factoredata2024admin"
password = "mdjdmliipo3^%^$5mkkm63"

fips_to_iso = {
    "AF": "AFG",
    "AX": "-",
    "AL": "ALB",
    "AG": "DZA",
    "AQ": "ASM",
    "AN": "AND",
    "AO": "AGO",
    "AV": "AIA",
    "AY": "ATA",
    "AC": "ATG",
    "AR": "ARG",
    "AM": "ARM",
    "AA": "ABW",
    "AT": "AUS",
    "AS": "AUS",
    "AU": "AUT",
    "AJ": "AZE",
    "BF": "BHS",
    "BA": "BHR",
    "FQ": "UMI",
    "BG": "BGD",
    "BB": "BRB",
    "BS": "REU",
    "BO": "BLR",
    "BE": "BEL",
    "BH": "BLZ",
    "BN": "BEN",
    "BD": "BMU",
    "BT": "BTN",
    "BL": "BOL",
    "BK": "BIH",
    "BC": "BWA",
    "BV": "BVT",
    "BR": "BRA",
    "IO": "IOT",
    "BX": "BRN",
    "BU": "BGR",
    "UV": "BFA",
    "BM": "MMR",
    "BY": "BDI",
    "CB": "KHM",
    "CM": "CMR",
    "CA": "CAN",
    "CV": "CPV",
    "CJ": "CYM",
    "CT": "CAF",
    "CD": "TCD",
    "CI": "CHL",
    "CH": "CHN",
    "KT": "CXR",
    "IP": "CLP",
    "CK": "CCK",
    "CO": "COL",
    "CN": "COM",
    "CG": "COD",
    "CF": "COG",
    "CW": "COK",
    "CR": "AUS",
    "CS": "CRI",
    "IV": "CIV",
    "HR": "HRV",
    "CU": "CUB",
    "UC": "CUW",
    "CY": "CYP",
    "EZ": "CZE",
    "DA": "DNK",
    "DX": "-",
    "DJ": "DJI",
    "DO": "DMA",
    "DR": "DOM",
    "EC": "ECU",
    "EG": "EGY",
    "ES": "SLV",
    "EK": "GNQ",
    "ER": "ERI",
    "EN": "EST",
    "ET": "ETH",
    "PJ": "-",
    "EU": "REU",
    "FK": "FLK",
    "FO": "FRO",
    "FJ": "FJI",
    "FI": "FIN",
    "FR": "FRA",
    "FG": "GUF",
    "FP": "PYF",
    "FS": "ATF",
    "GB": "GAB",
    "GA": "GMB",
    "GZ": "PSE",
    "GG": "GEO",
    "GM": "DEU",
    "GH": "GHA",
    "GI": "GIB",
    "GO": "REU",
    "GR": "GRC",
    "GL": "GRL",
    "GJ": "GRD",
    "GP": "GLP",
    "GQ": "GUM",
    "GT": "GTM",
    "GK": "GBR",
    "GV": "GIN",
    "PU": "GNB",
    "GY": "GUY",
    "HA": "HTI",
    "HM": "HMD",
    "HO": "HND",
    "HK": "HKG",
    "HQ": "UMI",
    "HU": "HUN",
    "IC": "ISL",
    "IN": "IND",
    "ID": "IDN",
    "IR": "IRN",
    "IZ": "IRQ",
    "EI": "IRL",
    "IM": "GBR",
    "IS": "ISR",
    "IT": "ITA",
    "JM": "JAM",
    "JN": "SJM",
    "JA": "JPN",
    "DQ": "UMI",
    "JE": "GBR",
    "JQ": "UMI",
    "JO": "JOR",
    "JU": "REU",
    "KZ": "KAZ",
    "KE": "KEN",
    "KQ": "UMI",
    "KR": "KIR",
    "KN": "PRK",
    "KS": "KOR",
    "KV": "-",
    "KU": "KWT",
    "KG": "KGZ",
    "LA": "LAO",
    "LG": "LVA",
    "LE": "LBN",
    "LT": "LSO",
    "LI": "LBR",
    "LY": "LBY",
    "LS": "LIE",
    "LH": "LTU",
    "LU": "LUX",
    "MC": "MAC",
    "MK": "MKD",
    "MA": "MDG",
    "MI": "MWI",
    "MY": "MYS",
    "MV": "MDV",
    "ML": "MLI",
    "MT": "MLT",
    "RM": "MHL",
    "MB": "MTQ",
    "MR": "MRT",
    "MP": "MUS",
    "MF": "MYT",
    "MX": "MEX",
    "FM": "FSM",
    "MQ": "UMI",
    "MD": "MDA",
    "MN": "MCO",
    "MG": "MNG",
    "MJ": "MNE",
    "MH": "MSR",
    "MO": "MAR",
    "MZ": "MOZ",
    "BM": "MMR",
    "WA": "NAM",
    "NR": "NRU",
    "BQ": "UMI",
    "NP": "NPL",
    "NL": "NLD",
    "NC": "NCL",
    "NZ": "NZL",
    "NU": "NIC",
    "NG": "NER",
    "NI": "NGA",
    "NE": "NIU",
    "NF": "NFK",
    "CQ": "MNP",
    "NO": "NOR",
    "MU": "OMN",
    "PK": "PAK",
    "PS": "PLW",
    "LQ": "UMI",
    "PM": "PAN",
    "PP": "PNG",
    "PF": "-",
    "PA": "PRY",
    "PE": "PER",
    "RP": "PHL",
    "PC": "PCN",
    "PL": "POL",
    "PO": "PRT",
    "RQ": "PRI",
    "QA": "QAT",
    "RE": "REU",
    "RO": "ROU",
    "RS": "RUS",
    "RW": "RWA",
    "TB": "BLM",
    "SH": "SHN",
    "SC": "KNA",
    "ST": "LCA",
    "RN": "MTQ",
    "SB": "SPM",
    "VC": "VCT",
    "WS": "WSM",
    "SM": "SMR",
    "TP": "STP",
    "SA": "SAU",
    "SG": "SEN",
    "RI": "SRB",
    "SE": "SYC",
    "SL": "SLE",
    "SN": "SGP",
    "NN": "SXM",
    "LO": "SVK",
    "SI": "SVN",
    "BP": "SLB",
    "SO": "SOM",
    "SF": "ZAF",
    "SX": "SGS",
    "OD": "SSD",
    "SP": "ESP",
    "PG": "-",
    "CE": "LKA",
    "SU": "SDN",
    "NS": "SUR",
    "SV": "SJM",
    "WZ": "SWZ",
    "SW": "SWE",
    "SZ": "CHE",
    "SY": "SYR",
    "TW": "TWN",
    "TI": "TJK",
    "TZ": "TZA",
    "TH": "THA",
    "TT": "TLS",
    "TO": "TGO",
    "TL": "TKL",
    "TN": "TON",
    "TD": "TTO",
    "TE": "UMI",
    "TS": "TUN",
    "TU": "TUR",
    "TX": "TKM",
    "TK": "TCA",
    "TV": "TUV",
    "UG": "UGA",
    "UP": "UKR",
    "AE": "ARE",
    "UK": "GBR",
    "US": "USA",
    "UY": "URY",
    "UZ": "UZB",
    "NH": "VUT",
    "VT": "VAT",
    "VE": "VEN",
    "VM": "VNM",
    "VI": "VGB",
    "VQ": "VIR",
    "-": "-",
    "-": "-",
    "YM": "YEM",
    "-": "-",
    "ZA": "ZMB",
    "ZI": "ZWE",
}

fips_to_name = {
    "AF": "Afghanistan",
    "AX": "Akrotiri",
    "AL": "Albania",
    "AG": "Algeria",
    "AQ": "American Samoa",
    "AN": "Andorra",
    "AO": "Angola",
    "AV": "Anguilla",
    "AY": "Antarctica",
    "AC": "Antigua and Barbuda",
    "AR": "Argentina",
    "AM": "Armenia",
    "AA": "Aruba",
    "AT": "Ashmore and Cartier Islands",
    "AS": "Australia",
    "AU": "Austria",
    "AJ": "Azerbaijan",
    "BF": "Bahamas, The",
    "BA": "Bahrain",
    "FQ": "Baker Island",
    "BG": "Bangladesh",
    "BB": "Barbados",
    "BS": "Bassas da India",
    "BO": "Belarus",
    "BE": "Belgium",
    "BH": "Belize",
    "BN": "Benin",
    "BD": "Bermuda",
    "BT": "Bhutan",
    "BL": "Bolivia",
    "BK": "Bosnia and Herzegovina",
    "BC": "Botswana",
    "BV": "Bouvet Island",
    "BR": "Brazil",
    "IO": "British Indian Ocean Territory",
    "BX": "Brunei",
    "BU": "Bulgaria",
    "UV": "Burkina Faso",
    "BM": "Burma",
    "BY": "Burundi",
    "CB": "Cambodia",
    "CM": "Cameroon",
    "CA": "Canada",
    "CV": "Cape Verde",
    "CJ": "Cayman Islands",
    "CT": "Central African Republic",
    "CD": "Chad",
    "CI": "Chile",
    "CH": "China",
    "KT": "Christmas Island",
    "IP": "Clipperton Island",
    "CK": "Cocos (Keeling) Islands",
    "CO": "Colombia",
    "CN": "Comoros",
    "CG": "Congo, Democratic Republic of the",
    "CF": "Congo, Republic of the",
    "CW": "Cook Islands",
    "CR": "Coral Sea Islands",
    "CS": "Costa Rica",
    "IV": "Cote d'Ivoire",
    "HR": "Croatia",
    "CU": "Cuba",
    "UC": "Curacao",
    "CY": "Cyprus",
    "EZ": "Czech Republic",
    "DA": "Denmark",
    "DX": "Dhekelia",
    "DJ": "Djibouti",
    "DO": "Dominica",
    "DR": "Dominican Republic",
    "EC": "Ecuador",
    "EG": "Egypt",
    "ES": "El Salvador",
    "EK": "Equatorial Guinea",
    "ER": "Eritrea",
    "EN": "Estonia",
    "ET": "Ethiopia",
    "PJ": "Etorofu, Habomai, Kunashiri, and Shikotan Islands",
    "EU": "Europa Island",
    "FK": "Falkland Islands (Islas Malvinas)",
    "FO": "Faroe Islands",
    "FJ": "Fiji",
    "FI": "Finland",
    "FR": "France",
    "FG": "French Guiana",
    "FP": "French Polynesia",
    "FS": "French Southern and Antarctic Lands",
    "GB": "Gabon",
    "GA": "Gambia, The",
    "GZ": "Gaza Strip",
    "GG": "Georgia",
    "GM": "Germany",
    "GH": "Ghana",
    "GI": "Gibraltar",
    "GO": "Glorioso Islands",
    "GR": "Greece",
    "GL": "Greenland",
    "GJ": "Grenada",
    "GP": "Guadeloupe",
    "GQ": "Guam",
    "GT": "Guatemala",
    "GK": "Guernsey",
    "GV": "Guinea",
    "PU": "Guinea-Bissau",
    "GY": "Guyana",
    "HA": "Haiti",
    "HM": "Heard Island and McDonald Islands",
    "HO": "Honduras",
    "HK": "Hong Kong",
    "HQ": "Howland Island",
    "HU": "Hungary",
    "IC": "Iceland",
    "IN": "India",
    "ID": "Indonesia",
    "IR": "Iran",
    "IZ": "Iraq",
    "EI": "Ireland",
    "IM": "Isle of Man",
    "IS": "Israel",
    "IT": "Italy",
    "JM": "Jamaica",
    "JN": "Jan Mayen",
    "JA": "Japan",
    "DQ": "Jarvis Island",
    "JE": "Jersey",
    "JQ": "Johnston Atoll",
    "JO": "Jordan",
    "JU": "Juan de Nova Island",
    "KZ": "Kazakhstan",
    "KE": "Kenya",
    "KQ": "Kingman Reef",
    "KR": "Kiribati",
    "KN": "Korea, North",
    "KS": "Korea, South",
    "KV": "Kosovo",
    "KU": "Kuwait",
    "KG": "Kyrgyzstan",
    "LA": "Laos",
    "LG": "Latvia",
    "LE": "Lebanon",
    "LT": "Lesotho",
    "LI": "Liberia",
    "LY": "Libya",
    "LS": "Liechtenstein",
    "LH": "Lithuania",
    "LU": "Luxembourg",
    "MC": "Macau",
    "MK": "Macedonia",
    "MA": "Madagascar",
    "MI": "Malawi",
    "MY": "Malaysia",
    "MV": "Maldives",
    "ML": "Mali",
    "MT": "Malta",
    "RM": "Marshall Islands",
    "MB": "Martinique",
    "MR": "Mauritania",
    "MP": "Mauritius",
    "MF": "Mayotte",
    "MX": "Mexico",
    "FM": "Micronesia, Federated States of",
    "MQ": "Midway Islands",
    "MD": "Moldova",
    "MN": "Monaco",
    "MG": "Mongolia",
    "MJ": "Montenegro",
    "MH": "Montserrat",
    "MO": "Morocco",
    "MZ": "Mozambique",
    "BM": "Myanmar",
    "WA": "Namibia",
    "NR": "Nauru",
    "BQ": "Navassa Island",
    "NP": "Nepal",
    "NL": "Netherlands",
    "NC": "New Caledonia",
    "NZ": "New Zealand",
    "NU": "Nicaragua",
    "NG": "Niger",
    "NI": "Nigeria",
    "NE": "Niue",
    "NF": "Norfolk Island",
    "CQ": "Northern Mariana Islands",
    "NO": "Norway",
    "MU": "Oman",
    "PK": "Pakistan",
    "PS": "Palau",
    "LQ": "Palmyra Atoll",
    "PM": "Panama",
    "PP": "Papua New Guinea",
    "PF": "Paracel Islands",
    "PA": "Paraguay",
    "PE": "Peru",
    "RP": "Philippines",
    "PC": "Pitcairn Islands",
    "PL": "Poland",
    "PO": "Portugal",
    "RQ": "Puerto Rico",
    "QA": "Qatar",
    "RE": "Reunion",
    "RO": "Romania",
    "RS": "Russia",
    "RW": "Rwanda",
    "TB": "Saint Barthelemy",
    "SH": "Saint Helena",
    "SC": "Saint Kitts and Nevis",
    "ST": "Saint Lucia",
    "RN": "Saint Martin",
    "SB": "Saint Pierre and Miquelon",
    "VC": "Saint Vincent and the Grenadines",
    "WS": "Samoa",
    "SM": "San Marino",
    "TP": "Sao Tome and Principe",
    "SA": "Saudi Arabia",
    "SG": "Senegal",
    "RI": "Serbia",
    "SE": "Seychelles",
    "SL": "Sierra Leone",
    "SN": "Singapore",
    "NN": "Sint Maarten",
    "LO": "Slovakia",
    "SI": "Slovenia",
    "BP": "Solomon Islands",
    "SO": "Somalia",
    "SF": "South Africa",
    "SX": "South Georgia and the Islands",
    "OD": "South Sudan",
    "SP": "Spain",
    "PG": "Spratly Islands",
    "CE": "Sri Lanka",
    "SU": "Sudan",
    "NS": "Suriname",
    "SV": "Svalbard",
    "WZ": "Swaziland",
    "SW": "Sweden",
    "SZ": "Switzerland",
    "SY": "Syria",
    "TW": "Taiwan",
    "TI": "Tajikistan",
    "TZ": "Tanzania",
    "TH": "Thailand",
    "TO": "Togo",
    "TL": "Tokelau",
    "TN": "Tonga",
    "TD": "Trinidad and Tobago",
    "TE": "Tromelin Island",
    "TS": "Tunisia",
    "TU": "Turkey",
    "TX": "Turkmenistan",
    "TK": "Turks and Caicos Islands",
    "TV": "Tuvalu",
    "UG": "Uganda",
    "UP": "Ukraine",
    "AE": "United Arab Emirates",
    "UK": "United Kingdom",
    "US": "United States",
    "UY": "Uruguay",
    "UZ": "Uzbekistan",
    "NH": "Vanuatu",
    "VE": "Venezuela",
    "VM": "Vietnam",
    "VQ": "Virgin Islands",
    "WQ": "Wake Island",
    "WF": "Wallis and Futuna",
    "WE": "West Bank",
    "WI": "Western Sahara",
    "YM": "Yemen",
    "ZA": "Zambia",
    "ZI": "Zimbabwe",
}

connection_string = "mssql+pyodbc://factoredata2024admin:mdjdmliipo3^%^$5mkkm63@factoredata2024.database.windows.net/dactoredata2024?driver=ODBC+Driver+18+for+SQL+Server"


@st.cache_data(ttl=60 * 60 * 12)  # Cache the data for 5 minutes (TTL: Time to Live)
def load_data():
    conn = connect(
        server=server,
        user=username,
        password=password,
        database=database,
    )

    query_goldstein = """
    SELECT [DATE], [pais], [y_pred], [y_real]
    FROM [events].[goldsteinPredictionsGold];
    """
    goldstein_data = pd.read_sql(query_goldstein, conn)

    query_tone = """
    SELECT [DATE], [Country], [y_pred], [y_real]
    FROM [gkg].[tonePredictionsGold];
    """
    tone_data = pd.read_sql(query_tone, conn)

    last_refreshed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return goldstein_data, tone_data, last_refreshed


goldstein_data, tone_data, last_refreshed = (
    load_data()
)  # Data will be cached and refreshed every 5 minutes


### DATA END ###

# Title and Header


# Add an Image (optional)
# st.image(
#     "https://via.placeholder.com/400", caption="Your Logo Here", use_column_width=True
# )


def presentation():
    st.markdown(
        """
# Factored Datathon 2024 - LatamFusion

## GDELT Analysis Dashboard

### A view of the world through data

See the world through data! This dashboard provides an overview of the GDELT dataset, which contains over 300 categories of events across the globe. Use the sidebar to navigate through different sections.

## Table of Contents 
- [Description](#description)
- [Features](#features)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Deployment](#Deployment)
- [About Us](#about-us)

## Description 

This project leverages the GDELT Project Dataset to generate critical insights that empower stakeholders to make data-driven decisions. Our web-based application enables users to monitor the current and projected situations of various countries, aiding in strategic planning and providing early warnings for potential risks. The solution is driven by AI, focusing on the analysis of two key indicators extracted from the GDELT dataset: Tone and GoldsteinScale. These metrics, when combined, offer a comprehensive view of a country's social, political, and economic stability as reflected in global news coverage, making them essential for evaluating regional stability and identifying areas of concern.

## Features 

- **Historical and Real-Time Data Visualization:** Explore and analyze time-series data from key indicators, both historical and streaming.
- **Indicator Evolution Forecasting:** Predict future trends of important indicators to anticipate potential risks and opportunities.
- **Automated Alerts:** Receive notifications when indicators surpass predefined thresholds, enabling proactive decision-making.
- **Insight Summarization:** Obtain concise summaries of significant insights drawn from global news coverage.
- **Interactive World Map:** Visualize insights across different regions with a comprehensive world map for an enhanced analytical experience.

## Project Structure

The project is composed of different directories used in different stages during the development of the project

* azfunc-streaming: files used to get the streaming data.
* crawler: files related to download the batch data from the Azure Datalake.
* databricks: ML models used in the pipelines of the project.
* eda notebook: exploratory data analysis useful to understand the nature of the data.
* web_solution: deploying of the solution in a web app using Taipy.

## Architecture
"""
    )

    st.image(
        "images/Architecture_LatamFusion.png",
        caption="Architecture of the LatamFusion Project",
    )

    st.markdown(
        """
## Deployment
Take a look of the latest version of our product here: https://latamfusionapp.azurewebsites.net/

# About us

Our team is composed of 4 members with different backgrounds and experiences. We are all passionate about data science and we are excited to share our findings with you. The name *LatamFusion* comes from the fact that we are all from different countries in Latin America and we are fusing our knowledge to create a great solution.

The team members are:

- [Hugo Vallejo](https://www.linkedin.com/in/hugo-r-vallejo-angulo/): Based on São Paulo, Brasil and originally from Caracas, Venezuela. Hugo is a PhD candidate in Artificial Intelligence at Universidade de São Paulo. Currently working as Data Engineer, Hugo contributed to the project by setting up the data pipeline and the ML model.

- [Agustín Rodríguez](https://www.linkedin.com/in/agustinnrodriguez/): Based on Buenos Aires, Argentina, and originally from the same city. Agustín is a Data Science & AI Enthusiast, currently working as Backend Developer. He contributed to the project by defining the business goal, exploring the data and developing analytics and ML solutions.

- [Jesús Castillo](https://www.linkedin.com/in/jes%C3%BAs-castillo/): Based on La Serena, Chile, and originally from the same city. Jesús is a Data Scientist. He comes from a background as a translator and interpreter. He is currently looking forward to expand his knowledge in the field of data science, particularly in LLMs, he contributed to the project by setting the time series model and improved it by hyperparameters optimization.

- [César Arroyo](https://www.linkedin.com/in/cesar-arroyo-cardenas): Based on Ciudad de México, México, and originally from Cartagena, Colombia. César has worked as Data Scientist and BI Developer. Currently working as Data Scientist, he contributed to the project by automating processes and setting up the web application.

"""
    )


def generate_alerts(df, country):
    """
    Generates alerts when y_pred is lower than the threshold in the future,
    with one alert per day, even if there are multiple records on the same day.

    Parameters:
    - df: DataFrame with columns ['DATE', 'Country', 'y_pred', 'y_real', 'y_pred_plus_one'].
    - Country: Name of the country for which to perform the analysis.
    """

    # Calculate threshold
    y_pred_mean = df["y_pred"].mean()
    y_pred_min = df["y_pred"].min()
    distance = y_pred_mean - y_pred_min
    threshold = (
        y_pred_mean - 0.3 * distance
    )  # Threshold at 30% distance from mean towards minimum

    # Filter future dates
    today = pd.Timestamp.today()
    df_future = df[df["DATE"] > today]

    # Identify alerts
    alerts = df_future[df_future["y_pred"] < threshold]

    # Group by date and take the first record of each day
    alerts = alerts.groupby("DATE").first().reset_index()

    # Display alerts in Streamlit
    if not alerts.empty:
        st.warning(f"Alerts for {country}: There are alerts on the following days:")
        # st.write(alerts[['DATE', 'y_pred', 'y_real']])
        st.table(alerts[["DATE", "y_pred"]])
    else:
        st.success(f"No alerts for {country}.")

def generate_past_alerts(df, country):
    """
    Generates alerts when y_pred is lower than the threshold in the future,
    with one alert per day, even if there are multiple records on the same day.

    Parameters:
    - df: DataFrame with columns ['DATE', 'Country', 'y_pred', 'y_real', 'y_pred_plus_one'].
    - Country: Name of the country for which to perform the analysis.
    """

    # Calculate threshold
    y_real_mean = df['y_real'].mean()
    y_real_min = df['y_real'].min()
    distance = y_real_mean - y_real_min
    threshold = y_real_mean - 0.70 * distance  # Threshold at 70% distance from mean towards minimum

    # Filter future dates
    today = pd.Timestamp.today()
    df_past = df[df['DATE'] < today]

    # Identify alerts
    past_alerts = df_past[df_past['y_real'] < threshold]

    # Group by date and take the first record of each day
    past_alerts = past_alerts.groupby('DATE').first().reset_index()

    # Display alerts in Streamlit
    if not past_alerts.empty:
        st.info(f"Past alerts for {country}: There were alerts on the following past days:")
        #st.write(alerts[['DATE', 'y_pred', 'y_real']])
        st.table(past_alerts[['DATE', 'y_real']])
    else:
        st.success(f"No past alerts for {country}.")

def goldsteinScale():
    st.header("Goldstein Scale")

    if goldstein_data is not None and not goldstein_data.empty:
        # Convert 'DATE' column to datetime
        goldstein_data["DATE"] = pd.to_datetime(goldstein_data["DATE"])

        # Filter data by country
        unique_countries = goldstein_data["pais"].unique()
        country = st.selectbox("Select Country", unique_countries)

        selected_name = fips_to_name.get(country, "Unknown Country")
        st.write(f"Selected Country: {selected_name}")

        # Filter data based on the selected country
        df_filtered = goldstein_data[goldstein_data["pais"] == country]
        df_filtered = df_filtered.sort_values(by="DATE")

        y_pred_mean = df_filtered["y_pred"].mean()

        # Calcular el valor mínimo de las predicciones
        y_pred_min = df_filtered["y_pred"].min()

        # Calcular la diferencia entre la media y el mínimo
        diff_mean_min = y_pred_mean - y_pred_min

        # Calcular el umbral al 30% de distancia desde la media
        threshold_30 = y_pred_mean - (diff_mean_min * 0.3)

        y_real_mean = df_filtered['y_real'].mean()

        # Calcular el valor mínimo de las predicciones
        y_real_min = df_filtered['y_real'].min()

        # Calcular la diferencia entre la media y el mínimo
        diff_real_mean_min = y_real_mean - y_real_min

        # Calcular el umbral al 30% de distancia desde la media
        threshold_70 = y_real_mean - (diff_real_mean_min * 0.7)

        # Plot the time series using Plotly graph_objects
        fig = go.Figure()

        # Add the 'y_pred' as a dashed line
        fig.add_trace(
            go.Scatter(
                x=df_filtered["DATE"],
                y=df_filtered["y_pred"],
                mode="lines",
                name="Predicted",
                line=dict(dash="dot"),  # Dashed line
            )
        )

        # Add the 'y_real' as a solid line
        fig.add_trace(
            go.Scatter(
                x=df_filtered["DATE"],
                y=df_filtered["y_real"],
                mode="lines",
                name="Real",
                line=dict(dash="solid"),  # Solid line
            )
        )

        fig.add_trace(go.Scatter(
            x=df_filtered['DATE'],
            y=[threshold_30] * len(df_filtered),
            mode='lines',
            name='30% Threshold from Pred Mean',
            line=dict(dash='dash', color='orange', width=1)
        ))

        fig.add_trace(go.Scatter(
            x=df_filtered['DATE'],
            y=[threshold_70] * len(df_filtered),
            mode='lines',
            name='70% Threshold from Real Mean',
            line=dict(dash='dash', color='purple', width=1)
        ))

        # Customize layout
        fig.update_layout(
            title=f"Time Series of the Goldstein Scale Average Index for: {selected_name}",
            xaxis_title="Date",
            yaxis_title="Values",
            legend_title="Type",
        )

        # Display plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Description
        st.write("""
        The Goldstein Scale Average Index is a numeric score ranging from -10 to +10, 
        representing the theoretical potential impact of events on a country's stability. 
        This index is based on news coverage of various events occurring in each country. 
        The type of event and its media coverage affect the stability of the country.
        """)

        generate_alerts(df_filtered, country)
        generate_past_alerts(df_filtered, country)
    else:
        st.warning("No data available to plot.")


### WORLD MAP #################################################################
def worldMap():
    st.header("World Map")

    # Filter the DataFrame to include only today's data (update date as needed)
    df = goldstein_data

    # Add a date selector for the map
    selected_date = st.date_input("Select Date for World Map", value=datetime(2024, 7, 30))

    st.write("""
    Please, select a date between 2023-09-14 and today
    """)

    df["iso_country"] = df["pais"].map(fips_to_iso)

    # Filter the DataFrame to include only data for the selected date
    df_today = df[df["DATE"].dt.date == selected_date]

    if not df_today.empty:
        # Create the interactive choropleth map
        fig_choropleth = px.choropleth(
            df_today,
            locations="iso_country",
            locationmode="ISO-3",
            color="y_real",
            color_continuous_scale="RdYlGn",  # Color scale
            title="World Map with Goldstein scale average index for each country",
        )

        # Display the choropleth map in Streamlit
        st.plotly_chart(fig_choropleth, use_container_width=True)
    else:
        st.warning("No data available for the specified date.")


    selected_date_future = st.date_input("Select Date for World Map", value=datetime(2024, 9, 20))
    
    st.write("""
    Please, select a date between today and the next 30 days (to view predictions)
    """)

    # Filter the DataFrame to include only data for the selected date
    df_tomorrow = df[df['DATE'].dt.date == selected_date_future]

    if not df_tomorrow.empty:
        # Create the interactive choropleth map
        fig_choropleth = px.choropleth(
            df_tomorrow, 
            locations='iso_country', 
            locationmode='ISO-3',
            color='y_pred',
            color_continuous_scale='Purples',  # Color scale
            title='World Map with Goldstein scale average index predictions for each country'
        )

        # Display the choropleth map in Streamlit
        st.plotly_chart(fig_choropleth, use_container_width=True)
    else:
        st.warning("No data available for the specified date.")


### PAGE SELECTOR ###

# Adding a Sidebar
st.sidebar.title("Navigation")
st.sidebar.write("Use this panel to navigate through different sections.")

# Sidebar options
option = st.sidebar.selectbox(
    "Choose a page:", ["Home", "Goldstein Scale by Country", "World Map"]
)

st.sidebar.write("Last updated on:", last_refreshed, "UTC")

### WORLD MAP #################################################################


# Customize home content based on user selection
if option == "Home":
    presentation()
# elif option == "Data":
#     st.write("Explore your data here.")
elif option == "Goldstein Scale by Country":
    # st.write("Check out your visualizations here.")
    goldsteinScale()

elif option == "World Map":
    worldMap()
# elif option == "Models":
#     st.write("Run your machine learning models here.")
