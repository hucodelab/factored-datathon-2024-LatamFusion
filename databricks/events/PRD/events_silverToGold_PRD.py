# Databricks notebook source
# MAGIC %md
# MAGIC ## Read the filtered DELTA file

# COMMAND ----------

# MAGIC %md
# MAGIC This way we can just read the filtered data from the datalake (in DBFS)

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

delta_path = "/mnt/silver/eventsSilver"
# Verify by reading the Delta table back (optional)
eventsSilver = spark.read.format("delta").load(delta_path)

# COMMAND ----------

# filtered by columns
delta_path = "/mnt/silver/eventsDAGASilver1"

eventsDAGASilver1 = eventsSilver \
    .select('DATE','Actor1CountryCode','ActionGeo_CountryCode','NumMentions','GoldsteinScale','AvgTone') \
        .na.drop(subset=['ActionGeo_CountryCode','GoldsteinScale','AvgTone','NumMentions'])

eventsDAGASilver1.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(delta_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read from silver layer

# COMMAND ----------

# filtered by columns
delta_path = "/mnt/silver/eventsDAGASilver1"

# COMMAND ----------

# Group the DataFrame by 'group_column'
weightedAvgGoldsteinToneGold = eventsDAGASilver1.groupBy("DATE","ActionGeo_CountryCode").agg(
    (F.sum(F.col("GoldsteinScale") * F.col("NumMentions")) / F.sum("NumMentions")).alias("GoldsteinScaleWA"),
    (F.sum(F.col("AvgTone") * F.col("NumMentions")) / F.sum("NumMentions")).alias("AvgToneWA")
)

# COMMAND ----------

storage_account_key = dbutils.secrets.get(scope="events", key="DataLakeKey")
storage_account_name = "factoredatathon2024"
container_name = "gold"

spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net",
    f"{storage_account_key}"
)

file_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/weightedAvgGoldsteinToneGold.csv"
weightedAvgGoldsteinToneGold = weightedAvgGoldsteinToneGold.coalesce(1)
weightedAvgGoldsteinToneGold.write.format("csv").mode("overwrite").option("mergeSchema", "true").option('header', 'true').save(file_path)

# COMMAND ----------

# print(df_read.count())
# 46 millions of rows
