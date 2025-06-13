#!/usr/bin/env python3

"""
ROS2 Node for loading a dataset from CSV, previewing it, and publishing as JSON.
Publishes the raw dataset to the preprocessor node.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import pandas as pd
import os
import io
import json

class Dataset_Loader(Node):
    """ROS2 Node for loading and publishing datasets."""

    def __init__(self):
        """
        Initialize the Dataset_Loader node, set up publisher and timer.
        """
        super().__init__("node_loader")
        self.publisher_ = self.create_publisher(String, "raw_dataset", 10)
        self.pose_subscriber_ack = self.create_subscription(String,"ack", self.recv_acknowledgement,10)
        self.timer = self.create_timer(1.0, self.data_loading)
        self.dataset_published = False
        self.get_logger().info("Loader node initialized. ")
        self.printed = False
        self.combined_msg = None

        # Prompt user for dataset name
        self.dataset_name = input("Enter the name of the dataset (without .csv): ").strip()
    
    def recv_acknowledgement(self, msg: String):
        """
        Callback for receiving acknowledgement from the preprocessor node.
        """
        parsed = json.loads(msg.data)
        if parsed["sender"] == 'node_preprocessor':
            if parsed["data"]:
                print("Received: True")
                self.dataset_published = True
            else:
                print("Received: False")
                self.dataset_published = False

    def publishing(self, combined_msg):
        """
        Publish the loaded dataset as a JSON string.
        """
        msg = String()
        msg.data = combined_msg
        self.publisher_.publish(msg)
        if not self.dataset_published:
            self.get_logger().info("Published dataset as JSON on 'raw_dataset'")

    def data_loading(self):
        """
        Timer callback for loading the dataset from CSV and publishing it.
        Also writes a preview of the dataset to a text file.
        """
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
                # Write a preview of the dataset to a text file for inspection
                dataset_path = os.path.join(
                                            "/home/can_ozkan/ros2_ws/src/ros_ml_implementation/ros_ml_implementation/data/",
                                            self.dataset_name + "_preview.txt"
                                            )
                with open(dataset_path, "w") as file:

                    file.write(f"---PREVIEW OF {self.dataset_name}: ---\n")
                    # Display header and first few rows
                    file.write("\n---------------------")
                    file.write(f"{self.dataset_name} Data Columns:\n")
                    file.write(', '.join(df.columns) + '\n')

                    # Display the first few rows of the dataset
                    file.write("\n---------------------")
                    (f"\n{self.dataset_name} Data Headers:\n")
                    file.write(df.head().to_string() + '\n')

                    # Get basic information
                    file.write("\n---------------------")
                    file.write(f"\n{self.dataset_name} Basic Info:\n")
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    file.write(buffer.getvalue() + '\n')

                    # Get statistical summary
                    file.write("\n---------------------")
                    file.write(f"\n{self.dataset_name} Statistic Summary:\n")
                    file.write(df.describe().to_string() + '\n')
                    
                    # Check for missing values
                    file.write("\n---------------------")
                    file.write(f"\n{self.dataset_name} Missing Values:\n")
                    file.write(df.isnull().sum().to_string() + '\n')
                self.printed = True

        except Exception as e:
            self.get_logger().error(f"Failed to load dataset: {e}")

def main(args=None):
    """
    Main entry point for the Dataset_Loader node.
    """
    rclpy.init(args=args)
    node = Dataset_Loader()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()