# spark_streaming_minibatch_cluster.py
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
import pandas as pd
import numpy as np
import csv
import io
from sklearn.cluster import MiniBatchKMeans

# ============
# Configuration
# ============
TRAINING_FILE = "iris_extended_train.csv"
# List of features used in clustering:
FEATURES = ['leaf_area_cm2', 'petal_area', 'petal_curvature_mm',
            'petal_length', 'petal_width', 'sepal_length', 'sepal_width']
N_CLUSTERS = 3
RANDOM_STATE = 42

# ===========================
# Initialize the clustering model
# ===========================
# Load the training data and extract the chosen features.
train_df = pd.read_csv(TRAINING_FILE)
X_train = train_df[FEATURES].values

# Initialize MiniBatchKMeans and do an initial partial fit using the training data.
model = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
model.partial_fit(X_train)
print("Initialized MiniBatchKMeans model using training data from", TRAINING_FILE)

# ===========================
# Set up PySpark Streaming Context
# ===========================
sc = SparkContext(appName="StreamingMiniBatchClustering")
# Use a batch interval of 5 seconds (adjust as needed)
ssc = StreamingContext(sc, 5)

# Create a DStream that listens on localhost:9999 (this is where your Go sender sends data)
lines = ssc.socketTextStream("localhost", 9999)

def process_rdd(time, rdd):
    """
    This function is called on each RDD (i.e. each batch of streamed records).
    It parses each CSV record, extracts the seven features, updates the model,
    and prints the assigned cluster for each record.
    """
    records = rdd.collect()
    if records:
        print("========= Batch Time:", str(time), "=========")
        features_batch = []  # To accumulate the features from this batch

        for line in records:
            # Parse the CSV record
            f = io.StringIO(line)
            reader = csv.reader(f)
            try:
                row = next(reader)
            except Exception as e:
                print("Error parsing line:", line, e)
                continue

            # Check if row has enough columns (we need at least 18 columns)
            if len(row) < 18:
                print("Incomplete record:", row)
                continue

            try:
                # Extract the seven features:
                #   leaf_area_cm2 (column index 17)
                #   petal_area (column index 8)
                #   petal_curvature_mm (column index 15)
                #   petal_length (column index 5)
                #   petal_width (column index 6)
                #   sepal_length (column index 3)
                #   sepal_width (column index 4)
                record_features = [
                    float(row[17]),  # leaf_area_cm2
                    float(row[8]),   # petal_area
                    float(row[15]),  # petal_curvature_mm
                    float(row[5]),   # petal_length
                    float(row[6]),   # petal_width
                    float(row[3]),   # sepal_length
                    float(row[4])    # sepal_width
                ]
                features_batch.append(record_features)
            except Exception as e:
                print("Error processing record:", row, e)

        if features_batch:
            X_batch = np.array(features_batch)
            # Update the model incrementally using the new batch.
            model.partial_fit(X_batch)
            # Predict the cluster labels for the batch
            clusters = model.predict(X_batch)
            for features_vector, cluster in zip(features_batch, clusters):
                print("Data:", features_vector, "-> Cluster:", cluster)

# Register the RDD processing function on the incoming DStream.
lines.foreachRDD(process_rdd)

# Start streaming and await termination.
ssc.start()
ssc.awaitTermination()
