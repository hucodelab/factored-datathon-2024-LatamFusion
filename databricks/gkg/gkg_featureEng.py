# Databricks notebook source
from pyspark.sql.functions import to_date, col, split, explode, to_date, array_contains, lit, when, count
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

# MAGIC %md
# MAGIC ### Silver layer: read OHE table

# COMMAND ----------

delta_path = "/mnt/silver/themesOHESilver"
# Verify by reading the Delta table back (optional)
df = spark.read.format("delta").load(delta_path)

# prepare the data for feature engineering
feat_eng = df.filter(col("date0") > "2023-01-01") \
    .drop("THEMES", "THEMES_ARRAY","") \
        .withColumn("TONE_ARRAY", split(df["TONE"], ",")) \
            .withColumn("NEGATIVE_TONE", col("TONE_ARRAY").getItem(2)) \ 
                .withColumn("POSITIVE_TONE", col("TONE_ARRAY").getItem(1)) \
                    .withColumn("NEGATIVE_TONE", col("NEGATIVE_TONE").cast("float")) \
                        .drop("TONE_ARRAY","TONE","date0")

# COMMAND ----------

def feature_engineering(df):

    # Step 1: Assemble all features into a single vector
    feature_columns = df.columns[:-2]  # the last column is the target
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = assembler.transform(df)

    # Step 2: RandomForestRegressor
    rf = RandomForestRegressor(featuresCol="features", labelCol="NEGATIVE_TONE", numTrees=100)

    # Step 3: Fit the model
    model = rf.fit(df)

    # Step 4: Extract feature importances
    importances = model.featureImportances.toArray()

    # Combine feature names with their importance scores
    feature_importances = sorted(zip(feature_columns, importances), key=lambda x: -x[1])
    return feature_importances

feature_importances = feature_engineering(feat_eng)

# COMMAND ----------

# Display top 10 most important features
for feature, importance in feature_importances[:10]:
    print(f"{feature}: {importance}")

# COMMAND ----------

def feature_engineering_assessment(df):
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

feature_engineering_assessment(feat_eng)
