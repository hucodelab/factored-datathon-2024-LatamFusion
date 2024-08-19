import pandas as pd

# from azure.storage.blob import BlobServiceClient


def process_df_to_plot():
    names = ["train", "test", "real"]

    datasets = [pd.read_csv(f"{name}.csv") for name in names]
    for dataset in datasets:
        dataset.columns = ["date", "goldstein"]
        dataset["date"] = pd.to_datetime(dataset["date"])

    return {name: dataset for name, dataset in zip(names, datasets)}


# def download_csv_from_blob(storage_account_url, container_name, blob_name, sas_token):
#     blob_service_client = BlobServiceClient(
#         account_url=storage_account_url, credential=sas_token
#     )
#     container_client = blob_service_client.get_container_client(container_name)
#     blob_client = container_client.get_blob_client(blob_name)

#     stream = blob_client.download_blob().readall()
#     return pd.read_csv(pd.compat.StringIO(stream.decode("utf-8")))


# def filter_data_by_country(df, country):
#     df["Date"] = pd.to_datetime(df["Date"])
#     return df[df["Country"] == country].sort_values("Date")
