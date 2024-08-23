# Databricks notebook source
from pyspark.sql.functions import to_date, col, split, explode, to_date, array_contains, lit, when, count, collect_list
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
import uuid

# COMMAND ----------

# account for landing files from https
storage_account_name = "factoredatathon2024"
storage_account_key = "wgbe0Fzs4W3dPNc35dp//uumz+SPDXVLLGu0mNaxTs2VLHCCPnD7u79PYt4mKeSFboqMRnZ+s+ez+ASty+k+sQ=="
container_name = "gkg"

spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net",
    f"{storage_account_key}"
)

file_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/*.csv"
df = spark.read.csv(file_path, sep="\t", header=True)

df = df.withColumn("date0", to_date("DATE", "yyyyMMdd"))
df = df.select("date0", "THEMES", "LOCATIONS", "TONE", "CAMEOEVENTIDS")
df = df.withColumn("THEMES_ARRAY", split(df["THEMES"], ";"))
df = df.withColumn("LOCATIONS_ARRAY", split(df["LOCATIONS"], "#"))
df = df.withColumn("CAMEOEVENTIDS_ARRAY", split(df["CAMEOEVENTIDS"], ","))
df = df.withColumn("countryCode", col("LOCATIONS_ARRAY").getItem(2))
df = df.withColumn("TONE_ARRAY", split(df["TONE"], ","))
df = df.withColumn("TONE_AVG", col("TONE_ARRAY").getItem(0))
df = df.withColumn("POLARITY", col("TONE_ARRAY").getItem(3).cast("float"))

#Filter by date and polarity to avoid highly polarized news:
df_reduced = df.filter((col("date0") > "2023-08-01") & (col("POLARITY") < 50))

# Define a function to generate UUIDs
def generate_uuid():
    return str(uuid.uuid4())

# Register the UDF
uuid_udf = udf(generate_uuid, StringType())

# Add UUID column to DataFrame
df_reduced = df_reduced.withColumn("uuid", uuid_udf())

# COMMAND ----------

dbfs_path_delta = "/mnt/silver/themesMappedSilver"
# Read the Delta table into a DataFrame
labels = spark.read.format("delta").load(dbfs_path_delta)

# COMMAND ----------

# Explode the themes into separate rows
df_exploded = df_reduced.withColumn("THEMES_EXPLODED2", explode(col("THEMES_ARRAY")))

# Join with df_mapping to get topics
df_with_topics = df_exploded.join(labels, df_exploded["THEMES_EXPLODED2"] == labels["THEMES_EXPLODED"], "left") \
                            .select(
                                        "uuid",
                                        "date0",
                                        'THEMES',
                                        'LOCATIONS',
                                        'TONE',
                                        'CAMEOEVENTIDS',
                                        'THEMES_ARRAY',
                                        'LOCATIONS_ARRAY',
                                        'CAMEOEVENTIDS_ARRAY',
                                        'countryCode',
                                        'TONE_ARRAY',
                                        'TONE_AVG',
                                        'POLARITY',  
                                        "THEMES_EXPLODED", 
                                        "Cluster_Name"
                                    )

# Group by id and aggregate topics into an
df_result = df_with_topics.groupBy(
                                "uuid",
                                "date0",
                                'countryCode',
                                'TONE_AVG'
                                   ) \
                .agg(collect_list("THEMES_EXPLODED").alias("THEMES2"),
                     collect_list("Cluster_Name").alias("Cluster_Name2")
                )

# # Show the result
# df_result.show(truncate=False)

# COMMAND ----------

# df_result.columns

# COMMAND ----------

# account for landing files from https
storage_account_name = "factoredatathon2024"
storage_account_key = "wgbe0Fzs4W3dPNc35dp//uumz+SPDXVLLGu0mNaxTs2VLHCCPnD7u79PYt4mKeSFboqMRnZ+s+ez+ASty+k+sQ=="
container_name = "silver"

# Configure Spark to use the storage account key
spark.conf.set(f"fs.azure.account.key.{storage_account_name}.dfs.core.windows.net", storage_account_key)

# Define the path to write the DataFrame
output_path = f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/gkgLabeledSilver1"

# Write DataFrame to ADLS as Delta (requires Delta Lake library)
df_result.write.format("delta").mode("overwrite").save(output_path)

# COMMAND ----------

# df_result.limit(5).show(truncate=False)

# COMMAND ----------


