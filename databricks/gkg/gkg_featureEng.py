# Databricks notebook source
from pyspark.sql.functions import to_date, col, split, explode, to_date, array_contains, lit, when, count

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

# COMMAND ----------

delta_path = "/mnt/silver/themesOHESilver"
# Verify by reading the Delta table back (optional)
df = spark.read.format("delta").load(delta_path)

# COMMAND ----------

# drop: THEMES, THEMES_ARRAY, ''
feat_eng = df.drop("THEMES", "THEMES_ARRAY","")
feat_eng = feat_eng.withColumn("TONE_ARRAY", split(df["TONE"], ","))
feat_eng = feat_eng.withColumn("NEGATIVE_TONE", col("TONE_ARRAY").getItem(2))
feat_eng = feat_eng.withColumn("POSITIVE_TONE", col("TONE_ARRAY").getItem(1))
feat_eng = feat_eng.drop("TONE_ARRAY","TONE")
feat_eng = feat_eng.withColumn("NEGATIVE_TONE", col("NEGATIVE_TONE").cast("float"))
feat_eng = feat_eng.filter(col("date0") > "2023-01-01")
feat_eng = feat_eng.drop("date0")

# COMMAND ----------

# feat_eng.count()
# feat_eng.filter(col("date0") > "2023-01-01").count()
# df.columns[-2:-1][0]

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
# from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import col

# Example DataFrame with 1000 one-hot encoded columns and 1 float column
df = feat_eng

# Step 1: Assemble all features into a single vector
feature_columns = df.columns[:-2]  # Assuming the last column is the target
# float_column = df.columns[-2:-1]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df = assembler.transform(df)

# Step 2: Choose a model - Regression or Classification
# Example with RandomForestRegressor
rf = RandomForestRegressor(featuresCol="features", labelCol="NEGATIVE_TONE", numTrees=100)

# Alternatively, for classification:
# rf = RandomForestClassifier(featuresCol="features", labelCol="float_column", numTrees=100)

# Step 3: Fit the model
model = rf.fit(df)

# Step 4: Extract feature importances
importances = model.featureImportances.toArray()

# Combine feature names with their importance scores
feature_importances = sorted(zip(feature_columns, importances), key=lambda x: -x[1])

# Display top 10 most important features
for feature, importance in feature_importances[:10]:
    print(f"{feature}: {importance}")

# COMMAND ----------

for feature, importance in feature_importances[:100]:
    print(f"{feature}: {importance}")

# COMMAND ----------

# Step 4: Make predictions on the same DataFrame (or use a test set if available)
predictions = model.transform(df.limit(10000))

# Step 5: Evaluate the model

# RMSE (Root Mean Squared Error)
rmse_evaluator = RegressionEvaluator(labelCol="NEGATIVE_TONE", predictionCol="prediction", metricName="rmse")
rmse = rmse_evaluator.evaluate(predictions)
print(f"RMSE: {rmse}")

# MAE (Mean Absolute Error)
mae_evaluator = RegressionEvaluator(labelCol="NEGATIVE_TONE", predictionCol="prediction", metricName="mae")
mae = mae_evaluator.evaluate(predictions)
print(f"MAE: {mae}")

# R-squared
r2_evaluator = RegressionEvaluator(labelCol="NEGATIVE_TONE", predictionCol="prediction", metricName="r2")
r2 = r2_evaluator.evaluate(predictions)
print(f"R-squared: {r2}")

# COMMAND ----------

# display(feat_eng)
