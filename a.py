#!/usr/bin/env python
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler
import sys
from pyspark.ml.evaluation import ClusteringEvaluator


# Create a SparkSession for offline training.
spark = SparkSession.builder \
    .appName("OfflineKMeansClustering") \
    .getOrCreate()

# -------------------------------
# 1. Load Extended IRIS Dataset
# -------------------------------
# Change the path to the location of your extended IRIS CSV file.
data_path = "iris_extended_train.csv"

# Read the dataset (assumes CSV with header and inferring schema)
iris_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(data_path)

# ------------------------------------------
# 2. Select Features for Clustering Training
# ------------------------------------------
# Use only the following features:
selected_features = ["leaf_area_cm2", "petal_area", "petal_curvature_mm",
                     "petal_length", "petal_width", "sepal_length", "sepal_width"]

training_df = iris_df.select(*selected_features)

# ------------------------------------
# 3. Assemble the Features into Vector
# ------------------------------------
assembler = VectorAssembler(inputCols=selected_features, outputCol="features")
training_df = assembler.transform(training_df)

# ---------------------------------------------------
# 4. (Optional) Standardize Features (recommended)
# ---------------------------------------------------
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withMean=True, withStd=True)
scalerModel = scaler.fit(training_df)
training_df = scalerModel.transform(training_df)

# --------------------------------------------------
# 5. Train the K-Means Model using Scaled Features
# --------------------------------------------------
# For example, set k=3 based on your analysis (e.g. elbow or silhouette method)
k = 3
kmeans = KMeans(featuresCol="scaledFeatures", predictionCol="prediction", k=k, seed=1)
kmeans_model = kmeans.fit(training_df)
print('*'*20, "end training", '*'*20)
# Optionally, print the cost (sum of squared distances)
# cost = kmeans_model.computeCost(training_df)
# print("Clustering cost: {:.2f}".format(cost))

# Make predictions (assign clusters to the training data)
predictions = kmeans_model.transform(training_df)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator(featuresCol="scaledFeatures", metricName="silhouette", distanceMeasure="squaredEuclidean")
silhouette_score = evaluator.evaluate(predictions)

print(f"Silhouette Score: {silhouette_score:.4f}")



# -----------------------------------------------------
# 6. Save the Trained Model, Scaler, and Assembler
# -----------------------------------------------------
# (Later these will be loaded by the streaming application.)
model_path      = "models/kmeans_model"
scaler_path     = "models/scaler_model"
assembler_path  = "models/feature_assembler"

kmeans_model.write().overwrite().save(model_path)
scalerModel.write().overwrite().save(scaler_path)
assembler.write().overwrite().save(assembler_path)

spark.stop()
