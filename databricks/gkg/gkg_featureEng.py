# Databricks notebook source
from pyspark.sql.functions import to_date, col, split, explode, to_date, array_contains, lit, when, count

# COMMAND ----------

delta_path = "/mnt/silver/themesOHESilver"
# Verify by reading the Delta table back (optional)
df = spark.read.format("delta").load(delta_path)

# COMMAND ----------

# drop: THEMES, THEMES_ARRAY, ''
feat_eng = df.drop("THEMES", "THEMES_ARRAY","")
feat_eng = feat_eng.withColumn("TONE_ARRAY", split(df["TONE"], ","))
feat_eng = feat_eng.withColumn("TONE1", col("TONE_ARRAY").getItem(0))
feat_eng = feat_eng.drop("TONE_ARRAY","TONE","date0")

# COMMAND ----------

print(len(feat_eng.columns))

# COMMAND ----------

display(feat_eng)
