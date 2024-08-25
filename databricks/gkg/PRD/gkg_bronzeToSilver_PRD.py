# Databricks notebook source
from pyspark.sql.functions import to_date, col, split, explode, to_date, array_contains, lit, when, count

# COMMAND ----------

# MAGIC %md
# MAGIC Read from bronze layer

# COMMAND ----------

# account for landing files from https
storage_account_name = "factoredatathon2024"
storage_account_key = dbutils.secrets.get(scope="events", key="DataLakeKey")
container_name = "gkg"

spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net",
    f"{storage_account_key}"
)
file_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/*.csv"
df = spark.read.csv(file_path, sep="\t", header=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Transfer into Silver layer

# COMMAND ----------

# convert into date format
df = df.withColumn("date0", to_date("DATE", "yyyyMMdd"))
df = df.select("date0", "THEMES", "TONE")
df = df.dropna(subset=["date0", "THEMES", "TONE"])
df = df.withColumn("THEMES_ARRAY", split(df["THEMES"], ";"))

# COMMAND ----------

# Step 1: Explode the array column
df_exploded = df.withColumn("THEMES_EXPLODED", explode(col("THEMES_ARRAY")))
# Step 2: Group by the exploded column and count occurrences
df_grouped = df_exploded.groupBy("THEMES_EXPLODED").agg(count("*").alias("count"))
# Step 3: Sort by count in descending order to find the most common values
df_sorted = df_grouped.orderBy(col("count").desc())

# COMMAND ----------

server = "factoredata2024.database.windows.net"
db = "dactoredata2024"
user = "factoredata2024admin"
password = dbutils.secrets.get(scope="events", key="ASQLPassword")

# JDBC connection properties
jdbc_url = f"jdbc:sqlserver://{server}:1433;database={db};user={user}@{db};password={password};encrypt=true;trustServerCertificate=false;hostNameInCertificate=*.database.windows.net;loginTimeout=30;"

connection_properties = {
    "user": f"{user}@{server}",
    "password": password,
    "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

# Table name in Azure SQL Database
table_name = "gkg.THEMES"

# Write DataFrame to Azure SQL Database
df_sorted.write.jdbc(url=jdbc_url, table=table_name, mode="overwrite", properties=connection_properties)

# COMMAND ----------

# Convert column to Python list
column_name = "THEMES_EXPLODED"
top1000_distinct_values = df_sorted.select(column_name).limit(1000).rdd.flatMap(lambda x: x).collect()

# COMMAND ----------

# Create one-hot encoded columns
for value in top1000_distinct_values:
    df = df.withColumn(value, when(array_contains(col("THEMES_ARRAY"), lit(value)), 1).otherwise(0))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Silver layer: One Hot Encoding

# COMMAND ----------

# MAGIC %md
# MAGIC We write the One Hot Encoded dataframe into silver layer

# COMMAND ----------

# Define the path where you want to save the Delta file in DBFS
delta_path = "/mnt/silver/themesOHESilver"

# Write the DataFrame as a Delta file
df = df.repartition(40)

df.write.format("delta") \
    .option("mergeSchema", "true") \
    .mode("overwrite") \
    .save(delta_path)
