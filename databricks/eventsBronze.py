# Databricks notebook source
# MAGIC %md
# MAGIC ## Read the filtered DELTA file

# COMMAND ----------

# MAGIC %md
# MAGIC This way we can just read the filtered data from the datalake (in DBFS)

# COMMAND ----------

delta_path = "/mnt/bronze/eventsBronze"
# Verify by reading the Delta table back (optional)
df_read = spark.read.format("delta").load(delta_path)

# COMMAND ----------

print(df_read.count())
# 46 millions of rows !!!
