#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import pandas as pd
import os
import json

class Dataset_Loader(Node):
    def __init__(self):
        super().__init__("node_loader")
        self.publisher_ = self.create_publisher(String, "raw_dataset", 10)
        self.pose_subscriber_ack = self.create_subscription(String,"ack", self.recv_acknowledgement,10)
        self.timer = self.create_timer(1.0, self.data_loading)
        self.dataset_published = False
        self.get_logger().info("Loader node initialized. ")
        self.printed = False
        self.combined_msg = None

        self.dataset_name = input("Enter the name of the dataset (without .csv): ").strip()
    
    def recv_acknowledgement(self, msg: String):
        parsed = json.loads(msg.data)
        if parsed["sender"] == 'node_preprocessor':
            if parsed["data"]:
                print("Received: True")
                self.dataset_published = True
            else:
                print("Received: False")
                self.dataset_published = False

    def publishing(self, combined_msg):
        msg = String()
        msg.data = combined_msg
        self.publisher_.publish(msg)
        if not self.dataset_published:
            self.get_logger().info("Published dataset as JSON on 'raw_dataset'")

    def data_loading(self):
        if self.dataset_published:
            self.publishing(self.combined_msg)
            return
            
        dataset_path = os.path.join(
            "/home/can_ozkan/ros2_ws/src/ros_ml_implementation/ros_ml_implementation/data/",
            self.dataset_name + ".csv"
        )
            
        try:
            df = pd.read_csv(dataset_path)
            self.get_logger().info(f"Loaded dataset '{self.dataset_name}' with shape {df.shape}")

            df_json = df.to_json(orient='split')  # Preserves structure and column names
            self.combined_msg = json.dumps({
                                        "dataset_name": self.dataset_name,
                                        "data": json.loads(df_json)  
                                    })
            
            self.publishing(self.combined_msg)
            
            if not self.printed:            
                print(f"\n ---PREVIEW OF {self.dataset_name}: ---")
                # Display header and first few rows
                print("\n---------------------")
                print(f"{self.dataset_name} Data Columns:\n")
                print(df.columns)

                # Display the first few rows of the dataset
                print("\n---------------------")
                (f"{self.dataset_name} Data Headers:\n")
                print(df.head())
                # Get basic information
                print("\n---------------------")
                print(f"{self.dataset_name} Basic Info:\n")
                print(df.info())

                # Get statistical summary
                print("\n---------------------")
                print(f"{self.dataset_name} Statistic Summary:\n")
                print(df.describe())
                
                # Check for missing values
                print("\n---------------------")
                print(f"{self.dataset_name} Missing Values:\n")
                print(df.isnull().sum())
                self.printed = True
                
                        

        except Exception as e:
            self.get_logger().error(f"Failed to load dataset: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = Dataset_Loader()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()