# Databricks notebook source
from pyspark.sql.functions import to_date, col, split, explode, to_date, array_contains, lit, when, count
from pyspark.sql import functions as F

# COMMAND ----------

# account for landing files from https
storage_account_name = "factoredatathon2024"
storage_account_key = "wgbe0Fzs4W3dPNc35dp//uumz+SPDXVLLGu0mNaxTs2VLHCCPnD7u79PYt4mKeSFboqMRnZ+s+ez+ASty+k+sQ=="
container_name = "gkg"

spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net",
    f"{storage_account_key}"
)

# COMMAND ----------

file_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/*.csv"
df = spark.read.csv(file_path, sep="\t", header=True)

# COMMAND ----------

df = df.withColumn("date0", to_date("DATE", "yyyyMMdd"))
df = df.select("date0", "THEMES", "LOCATIONS", "TONE", "CAMEOEVENTIDS")
df = df.withColumn("THEMES_ARRAY", split(df["THEMES"], ";"))
df = df.withColumn("LOCATIONS_ARRAY", split(df["LOCATIONS"], "#"))
df = df.withColumn("CAMEOEVENTIDS_ARRAY", split(df["CAMEOEVENTIDS"], ","))
df = df.withColumn("countryCode", col("LOCATIONS_ARRAY").getItem(2))
df = df.withColumn("TONE_ARRAY", split(df["TONE"], ","))
df = df.withColumn("TONE_AVG", col("TONE_ARRAY").getItem(0))

# COMMAND ----------

df_reduced = df.filter(col("date0") > "2023-08-01")

# COMMAND ----------


