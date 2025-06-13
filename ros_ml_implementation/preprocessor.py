#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import json


class Preprocessor(Node):
    def __init__(self):
        super().__init__('node_preprocessor')
        self.subscription = self.create_subscription(String, 'raw_dataset', self.listener_callback, 10)
        self.publisher_ack = self.create_publisher(String, "ack", 10)
        self.publisher_splitted = self.create_publisher(String, "splitted_dataset", 10)
        self.pose_subscriber_ack = self.create_subscription(String,"ack", self.recv_acknowledgement,10)
        self.splitted_dataset_published = False
        self.get_logger().info("Preprocessor node ready and listening on 'raw_dataset'")
        self.payload = None
        self.current_dataset = ""

    def sent_acknowledgement(self):
        msg = String()  
        msg.data = json.dumps({
                                "sender": self.get_name(),
                                "data": True
                            })
        self.publisher_ack.publish(msg)
    
    def recv_acknowledgement(self, msg: String):
        parsed = json.loads(msg.data)
        if parsed["sender"] == "node_model_trainer":
            if parsed["data"]:
                self.get_logger().info("Received: True")
                self.splitted_dataset_published = True
            else:
                self.get_logger().info("Received: False")
                self.splitted_dataset_published = False

    def publishing(self, payload):
        out_msg = String()
        out_msg.data = json.dumps(payload)
        self.publisher_splitted.publish(out_msg)
        if not self.splitted_dataset_published:
            self.get_logger().info(f"Published split dataset for '{self.current_dataset}'")
            self.splitted_dataset_published = True


    def listener_callback(self, msg: String):
        if self.splitted_dataset_published:
            parsed_msg = json.loads(msg.data)
            if self.current_dataset == parsed_msg["dataset_name"]:
                self.publishing(self.payload)
                return
            self.splitted_dataset_published = False
        
        try:
            parsed_msg = json.loads(msg.data)
            dataset_name = parsed_msg["dataset_name"]
            self.current_dataset = dataset_name
            df = pd.DataFrame(**parsed_msg["data"])
            self.get_logger().info(f"Received dataset '{dataset_name}' with shape: {df.shape}")
            self.sent_acknowledgement()

            # Preprocessing
            df.dropna(inplace=True)
            self.get_logger().info(f"After dropping NaNs: {df.shape}")

            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower, upper)
            self.get_logger().info("Outliers capped at 1st and 99th percentile.")

            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            self.get_logger().info("Features standardized.")

            # Feature/label split based on dataset
            if dataset_name == "boston_housing":
                X = df.drop(columns=["medv"])
                y = df["medv"]
            elif dataset_name == "new_height_weight":
                X = df[["Height"]] 
                y = df["Weight"]
            elif dataset_name == "human_brain":
                X = df[["Head Size(cm^3)"]]
                y = df["Brain Weight(grams)"]
            else:
                self.get_logger().error(f"Label extraction failed - unrecognized dataset: {dataset_name}")
                return

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Serialize split data into JSON
            self.payload = {
                "dataset_name": dataset_name,
                "X_train": X_train.to_json(orient='split'),
                "X_test": X_test.to_json(orient='split'),
                "y_train": y_train.to_json(orient='split'),
                "y_test": y_test.to_json(orient='split')
            }

            self.publishing(self.payload)

        except Exception as e:
            self.get_logger().error(f"Error during preprocessing: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = Preprocessor()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
