# Your Python version
FROM python:3.11-slim as taipy

# Web port of the application
EXPOSE 5000

# Create taipy user for security
RUN groupadd -r taipy && useradd -r -m -g taipy taipy
USER taipy

# Go to the dedicated folder and add the python corresponding folder in PATH
WORKDIR /home/taipy
ENV PATH="${PATH}:/home/taipy/.local/bin"

# Update pip
RUN python -m pip install --upgrade pip

# Install application's dependencies
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Install your application
COPY . .

# Start up command
# CMD ["python", "web_app.py"]
ENTRYPOINT [ "python", "web_app.py", "-P", "5000", "-H", "0.0.0.0", "--no-reloader" ]
