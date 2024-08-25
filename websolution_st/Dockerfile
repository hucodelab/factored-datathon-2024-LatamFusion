# app/Dockerfile

FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies for ODBC driver and curl
# RUN apt-get update && \
#     apt-get install -y \
#     curl \
#     gnupg2 \
#     unixodbc-dev \
#     apt-transport-https && \
#     curl https://packages.microsoft.com/keys/microsoft.asc | tee /etc/apt/trusted.gpg.d/microsoft.asc && \
#     curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list | tee /etc/apt/sources.list.d/mssql-release.list && \
#     apt-get update && \
#     ACCEPT_EULA=Y apt-get install -y msodbcsql18 && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# Copy the current directory content (including landing.py) to the container's /app directory
COPY . /app

# Install Python dependencies from the requirements.txt file (if it exists in the same directory)
RUN pip3 install -r requirements.txt

# Expose the default Streamlit port
EXPOSE 8501

# Add a healthcheck for the Streamlit service
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app using landing.py as the entry point
ENTRYPOINT ["streamlit", "run", "landing.py", "--server.port=8501", "--server.address=0.0.0.0"]
