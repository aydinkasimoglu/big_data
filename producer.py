from kafka import KafkaProducer
import pandas as pd
import json

data = pd.read_csv("preprocessed_data.csv")

data.drop(columns=["Timestamp"], inplace=True)

data["kafka_message"] = data.apply(lambda row: json.dumps(row.to_dict()), axis=1)

producer = KafkaProducer()

# Send messages to Kafka topic
try:
    for index, row in data.iterrows():
        producer.send("financialData", row["kafka_message"].encode())
        print(f"Sent message to financialData: {row['kafka_message']}")
except Exception as e:
    print(f"Error sending message: {e}")

producer.close()
