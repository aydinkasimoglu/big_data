from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, from_json, col
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession\
    .builder\
    .appName("FraudDetection")\
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.4")\
    .getOrCreate()

# Load the dataset
data = spark.read.csv("preprocessed_data.csv", inferSchema=True, header=True)

# Drop Timestamp and create features column
data = data.drop("Timestamp")
feature_columns = [col for col in data.columns if col != "Target"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data).withColumnRenamed("Target", "label")

(train_data, test_data) = data.randomSplit([0.8, 0.2], seed=42)

rf = RandomForestClassifier(numTrees=100)
model = rf.fit(train_data)

# Make predictions and evaluate the model
predictions = model.transform(test_data)
roc_auc = BinaryClassificationEvaluator().evaluate(predictions)

print(f"ROC AUC: {roc_auc}")

# Add a column for binary comparison between prediction and label
predictions = predictions.withColumn(
    "TP", expr("case when prediction = 1 AND label = 1 then 1 else 0 end")
)
predictions = predictions.withColumn(
    "TN", expr("case when prediction = 0 AND label = 0 then 1 else 0 end")
)
predictions = predictions.withColumn(
    "FP", expr("case when prediction = 1 AND label = 0 then 1 else 0 end")
)
predictions = predictions.withColumn(
    "FN", expr("case when prediction = 0 AND label = 1 then 1 else 0 end")
)

# Aggregate the counts for TP, TN, FP, and FN
metrics = predictions.agg(
    expr("sum(TP)").alias("TP"),
    expr("sum(TN)").alias("TN"),
    expr("sum(FP)").alias("FP"),
    expr("sum(FN)").alias("FN"),
).collect()[0]

TP, TN, FP, FN = metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"]

# Calculate metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if TP + FP != 0 else 0.0
recall = TP / (TP + FN) if TP + FN != 0 else 0.0
f1_score = (
    (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0.0
)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# Define the schema for the JSON message
schema = StructType(
    [
        StructField("Amount", DoubleType(), True),
        StructField("Merchant", DoubleType(), True),
        StructField("TransactionType", DoubleType(), True),
        StructField("Location", DoubleType(), True),
        StructField("Target", DoubleType(), True),
        StructField("Hour", DoubleType(), True),
        StructField("DayOfWeek", DoubleType(), True),
        StructField("Month", DoubleType(), True),
        StructField("DayOfMonth", DoubleType(), True),
    ]
)

# Read from Kafka
streaming_df = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("subscribe", "financialData")
    .load()
)

# Parse the JSON messages
parsed_df = (
    streaming_df.selectExpr("CAST(value AS STRING)")
    .select(from_json(col("value"), schema).alias("data"))
    .select("data.*")
)

# Process the data (e.g., predictions)
def process_batch(batch_df, epoch_id):
    # Assemble features
    features_df = assembler.transform(batch_df).withColumnRenamed("Target", "label")

    # Predict using the loaded model
    predictions = model.transform(features_df)
    predictions.select("Amount", "prediction").show(truncate=False)


# Start the streaming query
query = parsed_df.writeStream.foreachBatch(process_batch).start()
query.awaitTermination()