# Databricks notebook source
from pyspark.sql.functions import to_date, col, split, explode, to_date, array_contains, lit, when, count

# COMMAND ----------

# %pip install great_expectations

# COMMAND ----------

# account for landing files from https
storage_account_name = "factoredatathon2024"
storage_account_key = "wgbe0Fzs4W3dPNc35dp//uumz+SPDXVLLGu0mNaxTs2VLHCCPnD7u79PYt4mKeSFboqMRnZ+s+ez+ASty+k+sQ=="
container_name = "gkg"

# COMMAND ----------

spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net",
    f"{storage_account_key}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC Read from bronze layer

# COMMAND ----------

file_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/*.csv"
df = spark.read.csv(file_path, sep="\t", header=True)

# COMMAND ----------

# display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Transfer into Silver layer

# COMMAND ----------

df = df.withColumn("date0", to_date("DATE", "yyyyMMdd"))
df = df.select("date0", "THEMES", "TONE")
df = df.dropna(subset=["date0", "THEMES", "TONE"])
df = df.withColumn("THEMES_ARRAY", split(df["THEMES"], ";"))

# Explode the THEMES_ARRAY column
# df = df.withColumn("THEME", explode("THEMES_ARRAY"))

# COMMAND ----------

# Step 2: Get distinct values from the THEMES_ARRAY column
# distinct_values = df.selectExpr("explode(THEMES_ARRAY) as value").distinct().rdd.flatMap(lambda x: x).collect()

# COMMAND ----------

# Step 1: Explode the array column
df_exploded = df.withColumn("THEMES_EXPLODED", explode(col("THEMES_ARRAY")))

# Step 2: Group by the exploded column and count occurrences
df_grouped = df_exploded.groupBy("THEMES_EXPLODED").agg(count("*").alias("count"))

# Step 3: Sort by count in descending order to find the most common values
df_sorted = df_grouped.orderBy(col("count").desc())

# COMMAND ----------

spark.conf.set("credentials", "/Workspace/Shared/factored-datathon-gold-46ad9a4661b2.json")

# COMMAND ----------

# %fs pwd

# COMMAND ----------

dbutils.fs.ls("dbfs:/FileStore/shared_uploads/hvallejo.77@gmail.com/")

# COMMAND ----------

# Set the necessary configurations
spark.conf.set("credentialsFile", "dbfs:/FileStore/shared_uploads/hvallejo.77@gmail.com/factored_datathon_gold_46ad9a4661b2.json")
spark.conf.set("parentProject", "factored-datathon-gold")

# Define the necessary variables
project_id = "factored-datathon-gold"
dataset_id = "gkg"
table_id = "themes"

# Full table name
table_name = f"{project_id}:{dataset_id}.{table_id}"

# Write DataFrame to BigQuery
df_sorted.limit(5).write \
  .format("bigquery") \
  .option("table", table_name) \
  .option("parentProject", project_id) \
  .option("credentialsFile", "dbfs:/FileStore/shared_uploads/hvallejo.77@gmail.com/credentials.json") \
  .mode("overwrite") \
  .save()

# COMMAND ----------

server = "factoredata2024.database.windows.net"
db = "dactoredata2024"

# JDBC connection properties
jdbc_url = f"jdbc:sqlserver://factoredata2024.database.windows.net:1433;database=dactoredata2024;user=factoredata2024admin@factoredata2024;password=mdjdmliipo3^%^$5mkkm63;encrypt=true;trustServerCertificate=false;hostNameInCertificate=*.database.windows.net;loginTimeout=30;"

connection_properties = {
    "user": f"factoredata2024admin@{server}",
    "password": "mdjdmliipo3^%^$5mkkm63",
    "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

# Table name in Azure SQL Database
table_name = "gkg.THEMES"

# Write DataFrame to Azure SQL Database
df_sorted.write.jdbc(url=jdbc_url, table=table_name, mode="overwrite", properties=connection_properties)

# COMMAND ----------

# HERE WE WRITE ALL THE THEMES TO THE GOLD LAYER TO THEN PROCESS THEM USING NLP TO TAG EACH NEW
# storage_account_key = "wgbe0Fzs4W3dPNc35dp//uumz+SPDXVLLGu0mNaxTs2VLHCCPnD7u79PYt4mKeSFboqMRnZ+s+ez+ASty+k+sQ=="
# storage_account_name = "factoredatathon2024"
# container_name = "gold"

# spark.conf.set(
#     f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net",
#     f"{storage_account_key}"
# )

# file_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/gkg/themesSortedGold.csv"
# # weightedAvgGoldsteinToneGold = df_sorted.coalesce(1)
# df_sorted.coalesce(1).write.format("csv").mode("overwrite").option("mergeSchema", "true").option('header', 'true').save(file_path)

# COMMAND ----------

# Convert column to Python list
column_name = "THEMES_EXPLODED"
top1000_distinct_values = df_sorted.select(column_name).limit(1000).rdd.flatMap(lambda x: x).collect()

# COMMAND ----------

print(top1000_distinct_values)

# COMMAND ----------

# Step 3: Create one-hot encoded columns
for value in top1000_distinct_values:
    df = df.withColumn(value, when(array_contains(col("THEMES_ARRAY"), lit(value)), 1).otherwise(0))

# COMMAND ----------

# Define the path where you want to save the Delta file in DBFS
delta_path = "/mnt/silver/themesOHESilver"

# Write the DataFrame as a Delta file
df = df.repartition(40)

df.write.format("delta") \
    .option("mergeSchema", "true") \
    .mode("overwrite") \
    .save(delta_path)

# COMMAND ----------

delta_path = "/mnt/silver/themesOHESilver"
# Verify by reading the Delta table back (optional)
df = spark.read.format("delta").load(delta_path)

# COMMAND ----------

# len(df.columns)

# COMMAND ----------

# drop: THEMES, THEMES_ARRAY, ''
feat_eng = df.drop("THEMES", "THEMES_ARRAY","")
feat_eng = feat_eng.withColumn("THEMES_ARRAY", split(df["THEMES"], ";"))
