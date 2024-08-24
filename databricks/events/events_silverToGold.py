# Databricks notebook source
# MAGIC %md
# MAGIC ## Read the filtered DELTA file

# COMMAND ----------

# MAGIC %md
# MAGIC This way we can just read the filtered data from the datalake (in DBFS)

# COMMAND ----------

# Import required libraries
from pyspark.sql import functions as F

# COMMAND ----------

delta_path = "/mnt/silver/eventsSilver"
# Verify by reading the Delta table back (optional)
eventsSilver = spark.read.format("delta").load(delta_path)

# COMMAND ----------

# Filtered by columns
delta_path = "/mnt/silver/eventsDAGASilver1"

eventsDAGASilver1 = eventsSilver \
    .select('DATE','Actor1CountryCode','ActionGeo_CountryCode','NumMentions','GoldsteinScale','AvgTone') \
        .na.drop(subset=['ActionGeo_CountryCode','GoldsteinScale','AvgTone','NumMentions'])

eventsDAGASilver1.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(delta_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read from silver layer

# COMMAND ----------

# Filtered by columns
delta_path = "/mnt/silver/eventsDAGASilver1"
spark.read.format("delta").load(delta_path).show()

# COMMAND ----------

# Group the DataFrame by 'group_column'
weightedAvgGoldsteinToneGold = eventsDAGASilver1.groupBy("DATE","ActionGeo_CountryCode").agg(
    (F.sum(F.col("GoldsteinScale") * F.col("NumMentions")) / F.sum("NumMentions")).alias("GoldsteinScaleWA"),
    (F.sum(F.col("AvgTone") * F.col("NumMentions")) / F.sum("NumMentions")).alias("AvgToneWA")
)

# COMMAND ----------

# Set up Azure Blob Storage credentials and container details
storage_account_key = "wgbe0Fzs4W3dPNc35dp//uumz+SPDXVLLGu0mNaxTs2VLHCCPnD7u79PYt4mKeSFboqMRnZ+s+ez+ASty+k+sQ=="
storage_account_name = "factoredatathon2024"
container_name = "gold"

# Configure Spark to access Azure Blob Storage
spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net",
    f"{storage_account_key}"
)

# Define the file path in Blob Storage
file_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/weightedAvgGoldsteinToneGold.csv"

# Combine partitions into a single file
weightedAvgGoldsteinToneGold = weightedAvgGoldsteinToneGold.coalesce(1)

# Save the DataFrame as a CSV file to Azure Blob Storage
weightedAvgGoldsteinToneGold.write.format("csv").mode("overwrite").option("mergeSchema", "true").option('header', 'true').save(file_path)
