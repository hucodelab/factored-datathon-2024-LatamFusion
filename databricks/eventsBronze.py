# Databricks notebook source
# MAGIC %md
# MAGIC ## Read the filtered DELTA file

# COMMAND ----------

# MAGIC %md
# MAGIC This way we can just read the filtered data from the datalake (in DBFS)

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

delta_path = "/mnt/bronze/eventsBronze"
# Verify by reading the Delta table back (optional)
eventsBronze = spark.read.format("delta").load(delta_path)

# COMMAND ----------

# filtered by columns
delta_path = "/mnt/silver/eventsDAGASilver1"

eventsDAGASilver1 = eventsBronze \
    .select('DATE','Actor1CountryCode','ActionGeo_CountryCode','NumMentions','GoldsteinScale','AvgTone') \
        .na.drop(subset=['ActionGeo_CountryCode','GoldsteinScale','AvgTone','NumMentions'])

eventsDAGASilver1.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(delta_path)

# COMMAND ----------

from pyspark.sql import functions as F

# Group the DataFrame by 'group_column'
weightedAvgGoldsteinToneGold = eventsDAGASilver1.groupBy("DATE","ActionGeo_CountryCode").agg(
    (F.sum(F.col("GoldsteinScale") * F.col("NumMentions")) / F.sum("NumMentions")).alias("GoldsteinScaleWA"),
    (F.sum(F.col("AvgTone") * F.col("NumMentions")) / F.sum("NumMentions")).alias("AvgToneWA")
)

# delta_path = "/mnt/gold/weightedAvgGoldsteinToneGold"
# weightedAvgGoldsteinToneGold.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(delta_path)

# # Show the result
# weighted_avg_df.show()



# COMMAND ----------

storage_account_key = "wgbe0Fzs4W3dPNc35dp//uumz+SPDXVLLGu0mNaxTs2VLHCCPnD7u79PYt4mKeSFboqMRnZ+s+ez+ASty+k+sQ=="
storage_account_name = "factoredatathon2024"
container_name = "gold"

spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net",
    f"{storage_account_key}"
)

file_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/weightedAvgGoldsteinToneGold.csv"
weightedAvgGoldsteinToneGold = weightedAvgGoldsteinToneGold.coalesce(1)
weightedAvgGoldsteinToneGold.write.format("csv").mode("overwrite").option("mergeSchema", "true").save(file_path)

# COMMAND ----------

# from pyspark.sql.functions import col
# display(weighted_avg_df.orderBy(col('DATE').desc()))

# COMMAND ----------

# Assuming 'df' is your DataFrame
null_counts = eventsDAGASilver1.select(
    [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in eventsDAGASilver1.columns]
)

display(null_counts)

# COMMAND ----------



# COMMAND ----------

display(null_counts)

# COMMAND ----------

eventsBronze

# COMMAND ----------

display(eventsBronze)

# COMMAND ----------

eventsBronze.columns

# COMMAND ----------

print(df_read.count())
# 46 millions of rows !!!
