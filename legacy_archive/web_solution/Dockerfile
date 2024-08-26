# Base Python image
FROM python:3.11-slim as taipy

# Web port of the application
EXPOSE 5000

# Create taipy user for security
RUN groupadd -r taipy && useradd -r -m -g taipy taipy
USER taipy

# Go to the dedicated folder and add the python corresponding folder in PATH
WORKDIR /home/taipy
ENV PATH="${PATH}:/home/taipy/.local/bin"

# Switch to root user temporarily for driver installation
USER root

# Install dependencies for ODBC driver and curl
RUN apt-get update && \
    apt-get install -y \
    curl \
    gnupg2 \
    unixodbc-dev \
    apt-transport-https && \
    curl https://packages.microsoft.com/keys/microsoft.asc | tee /etc/apt/trusted.gpg.d/microsoft.asc && \
    curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list | tee /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql18 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Switch back to taipy user
USER taipy

# Update pip
RUN python -m pip install --upgrade pip

# Install application's dependencies
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Install your application
COPY . .

# Start up command
ENTRYPOINT [ "python", "web_app.py", "-P", "5000", "-H", "0.0.0.0", "--no-reloader" ]
