#!/usr/bin/env python
import socket as s
import threading
import time
from string import digits

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.feature import StandardScalerModel

from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.sql.functions import col, count, when, lit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import DataFrame


# Global list to hold incoming CSV lines and a lock for thread safety
data_batch = []
batch_lock = threading.Lock()

# --------------------------------------------------
# 1. Initialize Spark and Load Pre-Trained Models
# --------------------------------------------------
spark = SparkSession.builder.appName("StreamingKMeansClustering").getOrCreate()

# Paths (make sure these match your offline training)
model_path     = "models/kmeans_model"
scaler_path    = "models/scaler_model"
assembler_path = "models/feature_assembler"

# Load models and transformer(s)
kmeans_model = KMeansModel.load(model_path)
scalerModel  = StandardScalerModel.load(scaler_path)
assembler    = VectorAssembler.load(assembler_path)

# --------------------------------------------------
# 2. Define Schema and CSV Column Order
# --------------------------------------------------
# The full schema for the extended IRIS dataset.
schema = StructType([
    StructField("species", StringType(), True),
    StructField("elevation", DoubleType(), True),
    StructField("soil_type", StringType(), True),
    StructField("sepal_length", DoubleType(), True),
    StructField("sepal_width", DoubleType(), True),
    StructField("petal_length", DoubleType(), True),
    StructField("petal_width", DoubleType(), True),
    StructField("sepal_area", DoubleType(), True),
    StructField("petal_area", DoubleType(), True),
    StructField("sepal_aspect_ratio", DoubleType(), True),
    StructField("petal_aspect_ratio", DoubleType(), True),
    StructField("sepal_to_petal_length_ratio", DoubleType(), True),
    StructField("sepal_to_petal_width_ratio", DoubleType(), True),
    StructField("sepal_petal_length_diff", DoubleType(), True),
    StructField("sepal_petal_width_diff", DoubleType(), True),
    StructField("petal_curvature_mm", DoubleType(), True),
    StructField("petal_texture_trichomes_per_mm2", DoubleType(), True),
    StructField("leaf_area_cm2", DoubleType(), True),
    StructField("sepal_area_sqrt", DoubleType(), True),
    StructField("petal_area_sqrt", DoubleType(), True),
    StructField("area_ratios", DoubleType(), True),
])

# List of column names in CSV order
columns = ["species", "elevation", "soil_type", "sepal_length", "sepal_width",
           "petal_length", "petal_width", "sepal_area", "petal_area",
           "sepal_aspect_ratio", "petal_aspect_ratio", "sepal_to_petal_length_ratio",
           "sepal_to_petal_width_ratio", "sepal_petal_length_diff", "sepal_petal_width_diff",
           "petal_curvature_mm", "petal_texture_trichomes_per_mm2", "leaf_area_cm2",
           "sepal_area_sqrt", "petal_area_sqrt", "area_ratios"]

# --------------------------------------------------
# 3. CSV Parsing Function
# --------------------------------------------------
def parse_line(line):
    """
    Parse a CSV line into a tuple that matches the schema.
    Expects exactly 21 comma-separated values.
    """
    parts = line.strip().split(",")
    if len(parts) != len(columns):
        print("Invalid line (wrong number of columns):", line)
        return None
    try:
        # Parse string and numeric fields appropriately.
        species = parts[0]
        elevation = float(parts[1])
        soil_type = parts[2]
        sepal_length = float(parts[3])
        sepal_width  = float(parts[4])
        petal_length = float(parts[5])
        petal_width  = float(parts[6])
        sepal_area   = float(parts[7])
        petal_area   = float(parts[8])
        sepal_aspect_ratio = float(parts[9])
        petal_aspect_ratio = float(parts[10])
        sepal_to_petal_length_ratio = float(parts[11])
        sepal_to_petal_width_ratio  = float(parts[12])
        sepal_petal_length_diff = float(parts[13])
        sepal_petal_width_diff  = float(parts[14])
        petal_curvature_mm    = float(parts[15])
        petal_texture_trichomes_per_mm2 = float(parts[16])
        leaf_area_cm2         = float(parts[17])
        sepal_area_sqrt       = float(parts[18])
        petal_area_sqrt       = float(parts[19])
        area_ratios           = float(parts[20])
        return (species, elevation, soil_type, sepal_length, sepal_width,
                petal_length, petal_width, sepal_area, petal_area,
                sepal_aspect_ratio, petal_aspect_ratio, sepal_to_petal_length_ratio,
                sepal_to_petal_width_ratio, sepal_petal_length_diff, sepal_petal_width_diff,
                petal_curvature_mm, petal_texture_trichomes_per_mm2, leaf_area_cm2,
                sepal_area_sqrt, petal_area_sqrt, area_ratios)
    except Exception as e:
        print("Error parsing line:", line, "Error:", e)
        return None



def map_clusters_to_labels(predictions: DataFrame):
    """
    Map each predicted cluster to the majority class (species) using the current batch.
    Returns a dictionary {cluster_id: majority_label}.
    """
    majority_class_mapping = (
        predictions.groupBy("prediction", "species")
        .agg(count("*").alias("count"))
        .groupBy("prediction")
        .agg(expr("first(species) as majority_label"))
        .collect()
    )

    # Create mapping: {cluster_id: majority_class_label}
    return {row["prediction"]: row["majority_label"] for row in majority_class_mapping}

def calculate_metrics(predictions: DataFrame, label_col="species", pred_col="prediction"):
    """
    Calculate accuracy, precision, recall, and F1-score based on mapped labels.
    """
    # Step 1: Map cluster predictions to the majority class labels
    cluster_to_label_map = map_clusters_to_labels(predictions)

    # Step 2: Add a new column for the mapped predictions (majority label)
    mapped_predictions = predictions.withColumn(
        "mapped_prediction",
        expr(f"CASE { ' '.join([f'WHEN {pred_col} = {key} THEN {value}' for key, value in cluster_to_label_map.items()])} ELSE NULL END")
    )

    # Step 3: Calculate TP, FP, FN, TN using Spark SQL
    true_positives = mapped_predictions.filter(col("mapped_prediction") == col(label_col)).count()
    total = mapped_predictions.count()
    
    # Accuracy
    accuracy = true_positives / total if total > 0 else 0




# --------------------------------------------------
# 4. Custom Socket Server and Connection Handler
# --------------------------------------------------
def new_connection(conn: s.socket, addr):
    """
    Handle an individual client connection.
    Reads data in chunks, buffers incomplete lines, and
    appends complete CSV lines to the global data_batch.
    """
    try:
        print('[+] New Connection:', addr)
        buffer = ""
        while True:
            data = conn.recv(1024)  # Use a larger buffer size
            if not data:
                break
            buffer += data.decode('utf-8')
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                # If the client sends an exit command, break out.
                if line == '/exit':
                    conn.send(b'[-] Closing connection')
                    return
                # Otherwise, treat the line as CSV data.
                with batch_lock:
                    data_batch.append(line)
                print(f"[+] Received from {addr}: {line.split(',')[0]}")
                
    except Exception as e:
        print(f"[-] Error with connection {addr}: {e}")
    finally:
        conn.close()
        print("[-] Connection Closed:", addr)

def run_socket_server(host='localhost', port=9999):
    """
    Run a simple TCP server that accepts new connections
    and spawns a new thread for each client.
    """
    server_socket = s.socket(s.AF_INET, s.SOCK_STREAM)
    server_socket.setsockopt(s.SOL_SOCKET, s.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen()
    print(f"[+] Socket server listening on {host}:{port}")
    while True:
        conn, addr = server_socket.accept()
        threading.Thread(target=new_connection, args=(conn, addr), daemon=True).start()

# --------------------------------------------------
# 5. Batch Processing Loop (Every 1 Second)
# --------------------------------------------------
from pyspark.ml.clustering import KMeans

# Global storage for cumulative data and retrain trigger parameters
cumulative_data = []  # Store rows to retrain on full data
retrain_interval = 10  # Retrain the model after every 10 batches (can adjust this)
current_batch_count = 0  # Track batches processed
evaluator = ClusteringEvaluator(featuresCol="scaledFeatures", predictionCol="prediction", metricName="silhouette", distanceMeasure="squaredEuclidean")

def process_batches():
    """
    Every second, check if new CSV lines have arrived.
    If so, convert them into a DataFrame, run the Spark
    transformation pipeline (assembler, scaler, prediction),
    and print the results.
    """
    global current_batch_count, cumulative_data, kmeans_model

    while True:
        time.sleep(1)  # Process batches every 1 second
        with batch_lock:
            if not data_batch:
                continue
            # Copy current batch and clear the global list
            batch_lines = data_batch.copy()
            data_batch.clear()
        # Parse each CSV line into a row matching the schema
        rows = []
        for line in batch_lines:
            parsed = parse_line(line)
            if parsed is not None:
                rows.append(parsed)
        if not rows:
            continue
        # Create a DataFrame using the defined schema
        df = spark.createDataFrame(rows, schema)
        
        # -----------------------------------------------------
        # Process the DataFrame: assemble, scale, and predict
        # -----------------------------------------------------
        # (The assembler and scaler were saved during training.)
        df = assembler.transform(df)
        df = scalerModel.transform(df)
        
        # Predict cluster assignments
        predictions = kmeans_model.transform(df)
        
         # Map clusters to dominant class using the species label
        from pyspark.sql.functions import col, count, first
        from pyspark.ml.feature import StringIndexer

        cluster_to_class = (
            predictions.groupBy("prediction", "species")
            .count()
            .withColumn("rank", col("count").desc())  # Sort by descending count within groups
            .groupBy("prediction")
            .agg(
                first("species").alias("dominant_class"),  # Select the most frequent species as the dominant class
                first("count").alias("max_count")  # The corresponding count of the dominant class
            )
        )

        
        # Show selected columns: species, cluster prediction, and scaled features
        predictions.select("species", "prediction", "scaledFeatures").show(truncate=False)
        
        
        # ---------------------------
        # Evaluate clustering quality
        # ---------------------------
        # # Add evaluator for classification metrics
        # accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="true_label", predictionCol="prediction", metricName="accuracy")
        # recall_evaluator = MulticlassClassificationEvaluator(labelCol="true_label", predictionCol="prediction", metricName="weightedRecall")
        # f1_evaluator = MulticlassClassificationEvaluator(labelCol="true_label", predictionCol="prediction", metricName="f1")
        
        accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="prediction", metricName="accuracy")
        recall_evaluator = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="prediction", metricName="weightedRecall")
        f1_evaluator = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="prediction", metricName="f1")


        # Join and select necessary columns, rename species to true_label
        labeled_predictions = predictions.join(cluster_to_class, "prediction") \
        .select(predictions["prediction"], col("species").alias("true_label"), "scaledFeatures")

        # Cast prediction to double
        labeled_predictions = labeled_predictions.withColumn("prediction", col("prediction").cast("double"))

        # Index the true labels (species)
        label_indexer = StringIndexer(inputCol="true_label", outputCol="indexed_label")
        labeled_predictions = label_indexer.fit(labeled_predictions).transform(labeled_predictions)

        # # Rename the true species as the label column for evaluation
        # labeled_predictions = labeled_predictions.withColumnRenamed("species", "true_label")

        # ---------------------------
        # Evaluate Classification Metrics
        # ---------------------------
        accuracy = accuracy_evaluator.evaluate(labeled_predictions)
        recall = recall_evaluator.evaluate(labeled_predictions)
        f1_score = f1_evaluator.evaluate(labeled_predictions)

        print(f"[*] Accuracy: {accuracy:.4f}")
        print(f"[*] Recall: {recall:.4f}")
        print(f"[*] F1-score: {f1_score:.4f}")


        wait = False
        try:
            
            silhouette_score = evaluator.evaluate(predictions)
            print(f"[*] Silhouette score for current batch: {silhouette_score:.4f}")
        except: 
            wait = True
            print(f"[*] Silhouette found one cluster, waiting for more.")


       

        # ---------------------------
        # Retrain the KMeans model
        # ---------------------------
        cumulative_data.extend(rows)
        current_batch_count += 1

        if current_batch_count >= retrain_interval:
            print("\n\n\n\n[*] Retraining KMeans model with new data...\n\n\n\n")

            # Create a DataFrame using cumulative data
            cumulative_df = spark.createDataFrame(cumulative_data, schema)

            # Assemble and scale features for retraining
            cumulative_df = assembler.transform(cumulative_df)
            cumulative_df = scalerModel.transform(cumulative_df)
            
            # Evaluate silhouette score on cumulative data (optional)
            cumulative_predictions = kmeans_model.transform(cumulative_df)
            cumulative_silhouette = evaluator.evaluate(cumulative_predictions)
            print(f"[*] Silhouette score on cumulative data: {cumulative_silhouette:.4f}")


            # Retrain k-means with the updated data
            kmeans = KMeans(featuresCol="scaledFeatures", predictionCol="prediction", k=3, seed=1)
            kmeans_model = kmeans.fit(cumulative_df)
            print("\n\n\n\n\n")
            print("******************************************************************************")
            print("*****************[*] KMeans model retrained successfully.*********************")
            print("******************************************************************************")
            print("\n\n\n\n\n")

            # Reset counters
            if not wait:
                current_batch_count = 0
                cumulative_data = []  # Optional: Clear data if you want only recent data


# --------------------------------------------------
# 6. Main: Start the Server and Batch Processor
# --------------------------------------------------
if __name__ == "__main__":
    # Start the socket server in a daemon thread
    server_thread = threading.Thread(target=run_socket_server, args=('localhost', 9999), daemon=True)
    server_thread.start()
    
    # Continuously process incoming batches every second
    process_batches()
