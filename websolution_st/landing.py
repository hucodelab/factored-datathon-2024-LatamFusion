# import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine

# Title and Header
st.title("Welcome to My Streamlit Application")
st.header("Explore the Features and Analytics")

# Description
st.write("""
    This application is designed to provide insightful data analysis and visualization.
    Navigate through the side panel to explore different pages such as data uploads, charts, 
    and machine learning models. Enjoy your time exploring!
""")

# Add an Image (optional)
st.image(
    "https://via.placeholder.com/400", caption="Your Logo Here", use_column_width=True
)

# Adding a Sidebar
st.sidebar.title("Navigation")
st.sidebar.write("Use this panel to navigate through different sections.")

# Sidebar options
option = st.sidebar.selectbox(
    "Choose a page:", ["Home", "Data", "Visualizations", "Models"]
)

# Customize home content based on user selection
if option == "Home":
    st.write("You're on the Home page.")
elif option == "Data":
    st.write("Explore your data here.")
elif option == "Visualizations":
    st.write("Check out your visualizations here.")
elif option == "Models":
    st.write("Run your machine learning models here.")

# Azure SQL Database connection details
server = "factoredata2024.database.windows.net"
database = "dactoredata2024"
username = "factoredata2024admin"
password = "mdjdmliipo3^%^$5mkkm63"

connection_string = "mssql+pyodbc://factoredata2024admin:mdjdmliipo3^%^$5mkkm63@factoredata2024.database.windows.net/dactoredata2024?driver=ODBC+Driver+18+for+SQL+Server"

conn = create_engine(connection_string)

query = """
SELECT [DATE]
    ,[pais]
    ,[y_pred]
    ,[y_real]
FROM [events].[goldsteinPredictionsGold];
"""

goldstein_data = pd.read_sql(query, conn)

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


# Footer
st.write("Made with Streamlit")

# Optionally, add a contact form or external link
st.markdown("[Contact Us](mailto:someone@example.com)")
