"""
Azure Function to fetch GDELT data and upload to Azure Blob Storage.

It is triggered by a timer trigger and downloads the GDELT events and GKG
files for the previous day. The files are then uploaded to Azure Blob
Storage.


Notes
-----

When deploying this function app you should always make sure to set this
environment variable in the Azure Function App settings:
`WEBSITES_ENABLE_APP_SERVICE_STORAGE` = False

This variable controls whether the `/home/` directory is persisted across
function app restarts and shared across duplicated instances when scaling
up. Since for our case we are just downloading and uploading files via a
time trigger, we don't need this functionality.

"""

import datetime
import logging
import os
import tempfile
import zipfile

import azure.functions as func
import requests
from azure.storage.blob import BlobServiceClient

app = func.FunctionApp()

# Set up environment variables for the connection string and container name
STORAGE_CONNECTION_STRING_EVENTS = os.getenv("AZURE_STORAGE_CONNECTION_STRING_EVENTS")
BLOB_CONTAINER_EVENTS = os.getenv("BLOB_CONTAINER_EVENTS")
BLOB_SERVICE_CLIENT_EVENTS = BlobServiceClient.from_connection_string(
    STORAGE_CONNECTION_STRING_EVENTS
)

STORAGE_CONNECTION_STRING_GKG = os.getenv("AZURE_STORAGE_CONNECTION_STRING_GKG")
BLOB_CONTAINER_GKG = os.getenv("BLOB_CONTAINER_GKG")
BLOB_SERVICE_CLIENT_GKG = BlobServiceClient.from_connection_string(
    STORAGE_CONNECTION_STRING_GKG
)


# Temp directory, base URL for GDELT data, and Blob Service Client
TEMP_DIR = tempfile.gettempdir()
BASE_URL_EVENTS = "http://data.gdeltproject.org/events"
BASE_URL_GKG = "http://data.gdeltproject.org/gkg"


def download_file(url, filename):
    """
    Downloads a file from a URL.

    Arguments
    ---------
    url : str
        The URL to download the file from.
    filename : str
        The name of the file to save the downloaded content to.

    Returns
    -------
    bool
        True if the download was successful, False otherwise.

    """
    filepath = os.path.join(TEMP_DIR, filename)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return True
    else:
        logging.error(f"Failed to download {url}")
        return False


def unzip(file_path):
    """
    Unzips a file.

    Arguments
    ---------
    file_path : str
        The path to the zipped file.

    Returns
    -------
    None

    """
    try:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(TEMP_DIR)
        logging.info(f"Unzipped {file_path}.")
    except Exception as e:
        logging.error(f"Failed to unzip {file_path}. Error: {e}")


def upload_file_to_blob(filename: str, type: str) -> None:
    """
    Uploads a file to Azure Blob Storage.

    Arguments
    ---------
    file_path : str
        The local path to the file to upload.
    filename : str
        The name of the blob to create in the container.

    Returns
    -------
    None

    """

    match type:
        case "events":
            BLOB_SERVICE_CLIENT = BLOB_SERVICE_CLIENT_EVENTS
            container = BLOB_CONTAINER_EVENTS
        case "gkg":
            BLOB_SERVICE_CLIENT = BLOB_SERVICE_CLIENT_GKG
            container = BLOB_CONTAINER_GKG

    try:
        blob_client = BLOB_SERVICE_CLIENT.get_blob_client(
            container=container, blob=filename
        )
        filepath = os.path.join(TEMP_DIR, filename)

        logging.info(f"Uploading {filename} to blob storage...")

        with open(filepath, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        logging.info(f"Uploaded {filename} to blob storage.")
    except Exception as e:
        logging.error(f"Failed to upload {filename} to blob storage. Error: {e}")


def execute_complete_extract_load(file_url: str, filename: str, type: str) -> None:
    """
    Execute the complete EL process for a given file.

    Arguments
    ---------
    file_url : str
        The URL of the file to download.
    filename : str
        The name of the file to save the downloaded content to and to upload.

    """

    try:
        if download_file(file_url, filename):
            logging.info(f"Downloaded {filename}.")
            unzip(os.path.join(TEMP_DIR, filename))

            filename = filename[:-4]
            upload_file_to_blob(filename, type)

            logging.info(f"Cleaning up {os.path.join(TEMP_DIR, filename)}.")
            os.remove(os.path.join(TEMP_DIR, filename))
            logging.info(f"Cleaning up {os.path.join(TEMP_DIR, f'{filename}.zip')}.")
            os.remove(os.path.join(TEMP_DIR, f"{filename}.zip"))
    except Exception as e:
        logging.error(f"Failed to process {filename}.")
        logging.error(f"Error: {e}")


@app.function_name("streaming_gdelt_events")
@app.schedule(
    schedule="0 */5 * * * *",  # Uncomment for testing every minute
    # schedule="0 30 8 */1 * *",
    arg_name="streamingcron",
    run_on_startup=True,  # Always False in production. True for testing only.
    use_monitor=False,
)
def streaming_gdelt_events(streamingcron) -> None:
    logging.info(
        "Azure Function triggered to fetch GDELT events and upload to blob storage."
    )

    today = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(days=2)
    date = today.strftime("%Y%m%d")
    # Construct file name and URL for the GDELT file
    file_name_events = f"{date}.export.CSV.zip"
    file_url_events = f"{BASE_URL_EVENTS}/{file_name_events}"

    logging.info(f"Downloading {file_url_events}")

    execute_complete_extract_load(file_url_events, file_name_events, type="events")

    logging.info("Azure Function completed.")


@app.function_name("streaming_gdelt_gkg")
@app.schedule(
    schedule="0 */5 * * * *",  # Uncomment for testing every minute
    # schedule="0 30 8 */1 * *",
    arg_name="streamingcron",
    run_on_startup=True,  # Always False in production. True for testing only.
    use_monitor=False,
)
def streaming_gdelt_gkg(streamingcron) -> None:
    logging.info(
        "Azure Function triggered to fetch GDELT GKG and upload to blob storage."
    )

    today = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(days=2)
    date = today.strftime("%Y%m%d")
    # Construct file name and URL for the GDELT file
    file_name_gkg = f"{date}.gkg.csv.zip"
    file_url_gkg = f"{BASE_URL_GKG}/{file_name_gkg}"

    logging.info(f"Downloading {file_url_gkg}")

    execute_complete_extract_load(file_url_gkg, file_name_gkg, type="gkg")

    logging.info("Azure Function completed.")
