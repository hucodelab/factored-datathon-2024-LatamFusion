# Databricks notebook source
# account for landing files from https
storage_account_name = "factoredatathon2024"
storage_account_key = "wgbe0Fzs4W3dPNc35dp//uumz+SPDXVLLGu0mNaxTs2VLHCCPnD7u79PYt4mKeSFboqMRnZ+s+ez+ASty+k+sQ=="
container_name = "silver"

# Configure Spark to use the storage account key
spark.conf.set(f"fs.azure.account.key.{storage_account_name}.dfs.core.windows.net", storage_account_key)

# Define the path to write the DataFrame
file_path = f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/gkgLabeledSilver1"


# Read the Parquet files
df = spark.read.format("delta").load(file_path)

# COMMAND ----------

df.show()

# COMMAND ----------


