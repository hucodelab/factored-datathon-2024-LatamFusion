import datetime
import logging
import os

import azure.functions as func
import requests
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

app = func.FunctionApp()

# Set up environment variables for the connection string and container name
STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")


# Base URL for GDELT events
BASE_URL = "http://data.gdeltproject.org/events"


def download_file(url, filename):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return True
    else:
        logging.error(f"Failed to download {url}")
        return False


def upload_to_blob(file_path, blob_name):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(
            STORAGE_CONNECTION_STRING
        )
        blob_client = blob_service_client.get_blob_client(
            container=BLOB_CONTAINER_NAME, blob=blob_name
        )

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        logging.info(f"Uploaded {blob_name} to blob storage.")
    except Exception as e:
        logging.error(f"Failed to upload {blob_name} to blob storage. Error: {e}")


@app.schedule(
    schedule="0 0 3,9,15,21 * * *",
    arg_name="streamingcron",
    run_on_startup=True,
    use_monitor=False,
)
def streaming_gdelt(streamingcron) -> None:
    logging.info(
        "Azure Function triggered to fetch GDELT events and upload to blob storage."
    )

    # Calculate the last 3 days
    today = datetime.datetime.now(tz=datetime.timezone.utc)
    dates = [
        (today - datetime.timedelta(days=i)).strftime("%Y%m%d") for i in range(1, 4)
    ]

    for date in dates:
        # Construct file name and URL for the GDELT file
        file_name = f"{date}.export.CSV.zip"
        file_url = f"{BASE_URL}/{file_name}"

        # Download the file
        try:
            if download_file(file_url, file_name):
                # Upload the file to Azure Blob Storage
                upload_to_blob(file_name, file_name)

                # Clean up the local file after upload
                os.remove(file_name)
        except Exception as e:
            logging.error(f"Failed to process {file_name}.")
            logging.error(f"Error: {e}")
