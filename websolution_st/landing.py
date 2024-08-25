# import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine

### DATA LOADING ###
# Azure SQL Database connection details
server = "factoredata2024.database.windows.net"
database = "dactoredata2024"
username = "factoredata2024admin"
password = "mdjdmliipo3^%^$5mkkm63"

connection_string = "mssql+pyodbc://factoredata2024admin:mdjdmliipo3^%^$5mkkm63@factoredata2024.database.windows.net/dactoredata2024?driver=ODBC+Driver+18+for+SQL+Server"

conn = create_engine(connection_string)

query_goldstein = """
SELECT [DATE]
    ,[pais]
    ,[y_pred]
    ,[y_real]
FROM [events].[goldsteinPredictionsGold];
"""

goldstein_data = pd.read_sql(query_goldstein, conn)

query_tone = """
SELECT 
     [DATE]
    ,[Country]
    ,[y_pred]
    ,[y_real]
FROM [gkg].[tonePredictionsGold];
"""

tone_data = pd.read_sql(query_tone, conn)

### DATA END ###

# Title and Header
st.title("GDELT Analysis Dashboard")
st.header("A view of the world through data")

# Description
st.write("""
See the world through data! This dashboard provides an overview of the GDELT dataset, which contains over 300 categories of events across the globe. Use the sidebar to navigate through different sections.
""")

# Add an Image (optional)
# st.image(
#     "https://via.placeholder.com/400", caption="Your Logo Here", use_column_width=True
# )


def presentation():
    st.markdown(
        """
# Factored Datathon 2024 - LatamFusion*

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

![Architecture](/images/Architecture_LatamFusion.png)

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


def visualizations():
    st.write(goldstein_data.head())

    if goldstein_data is not None and not goldstein_data.empty:
        st.write("Data Preview:", goldstein_data.head())

        # Convert 'DATE' column to datetime
        goldstein_data["DATE"] = pd.to_datetime(goldstein_data["DATE"])

        # Filter data by country
        unique_countries = goldstein_data["pais"].unique()
        country = st.selectbox("Select Country", unique_countries)

        # Filter data based on the selected country
        df_filtered = goldstein_data[goldstein_data["pais"] == country]

        # Plot the time series using Plotly
        fig = px.line(
            df_filtered,
            x="DATE",
            y=["y_pred", "y_real"],
            labels={"value": "Values", "variable": "Type"},
            title=f"Time Series for {country}",
        )

        # Display plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available to plot.")


### PAGE SELECTOR ###

# Adding a Sidebar
st.sidebar.title("Navigation")
st.sidebar.write("Use this panel to navigate through different sections.")

# Sidebar options
option = st.sidebar.selectbox("Choose a page:", ["Home", "Visualizations"])


# Customize home content based on user selection
if option == "Home":
    st.write("You're on the Home page.")
    presentation()
# elif option == "Data":
#     st.write("Explore your data here.")
elif option == "Visualizations":
    st.write("Check out your visualizations here.")
    visualizations()
# elif option == "Models":
#     st.write("Run your machine learning models here.")
