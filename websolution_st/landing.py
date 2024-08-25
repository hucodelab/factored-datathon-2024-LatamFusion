import streamlit as st

# Title and Header
st.title("GDELT Risk Analysis")
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
