import os

import pandas as pd
import plotly.express as px
import pymssql  # Or use pyodbc if preferred
import streamlit as st

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
server = os.getenv("SQL_SERVER")
database = os.getenv("SQL_SERVER_DATABASE")
username = os.getenv("SQL_SERVER_ADMIN")
password = os.getenv("SQL_SERVER_PASSWORD")


# Connect to Azure SQL
def get_data_from_azure():
    try:
        conn = pymssql.connect(
            server=server, user=username, password=password, database=database
        )
        query = """
                SELECT [fecha]
                    ,[pais]
                    ,[y_pred]
                    ,[y_real]
                FROM [events].[goldsteinPredictionsGold]
                order by fecha asc;
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        return None


# Fetch data from Azure SQL
st.title("Time Series from Azure SQL Database")

df = get_data_from_azure()

if df is not None and not df.empty:
    st.write("Data Preview:", df.head())

    # Convert 'fecha' column to datetime
    df["fecha"] = pd.to_datetime(df["fecha"])

    # Filter data by country
    unique_countries = df["pais"].unique()
    country = st.selectbox("Select Country", unique_countries)

    # Filter data based on the selected country
    df_filtered = df[df["pais"] == country]

    # Plot the time series using Plotly
    fig = px.line(
        df_filtered,
        x="fecha",
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
