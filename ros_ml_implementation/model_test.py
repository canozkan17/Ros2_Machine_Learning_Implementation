#!/usr/bin/env python3

"""
ROS2 Node for testing a trained machine learning model.
Receives test data, loads the trained model, evaluates predictions, and saves results/visualizations.
Publishes acknowledgements to coordinate with other nodes.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import json
import pandas as pd
import pickle
import os
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class ModelTester(Node):
    """ROS2 Node for testing a trained ML model and reporting results."""

    def __init__(self):
        """
        Initialize the ModelTester node, set up subscriptions and publishers.
        """
        super().__init__('node_model_tester')
        self.subscription = self.create_subscription(
            String,
            'splitted_dataset',
            self.listener_callback,
            10
        )
        self.publisher_ack = self.create_publisher(String, "ack", 10)
        self.pose_subscriber_model = self.create_subscription(Bool,"training_complete", self.recv_model_ready,10)
        self.test_complete = False
        self.model_ready = False
        self.current_dataset = ""
        self.get_logger().info("Tester node initialized and listening on 'splitted_dataset'")

    def send_acknowledgement(self):
        """
        Publish an acknowledgement message to notify other nodes.
        """
        ack_msg = String()  
        ack_msg.data = json.dumps({
                                "sender": self.get_name(),
                                "data": True
                            })
        self.publisher_ack.publish(ack_msg)
    
    def recv_model_ready(self, msg: Bool):
        """
        Callback for receiving notification that the model is ready for testing.
        """
        if msg.data:
            self.model_ready = True

    def listener_callback(self, msg: String):
        """
        Callback for receiving test data, loading the model, evaluating, and saving results.
        """
        if not self.model_ready:
            return
        if self.test_complete:
            parsed_msg = json.loads(msg.data)
            if self.current_dataset == parsed_msg["dataset_name"]:
                self.send_acknowledgement() 
                return
            self.test_complete = False

        try:
            parsed_msg = json.loads(msg.data)
            dataset_name = parsed_msg["dataset_name"]
            self.current_dataset = dataset_name

            self.get_logger().info(f"Received test data for dataset '{dataset_name}'")

            # Deserialize test data
            X_test = pd.read_json(parsed_msg["X_test"], orient='split')
            y_test = pd.read_json(parsed_msg["y_test"], orient='split', typ='series')

            # Load trained model
            model_path = f"/home/can_ozkan/ros2_ws/src/ros_ml_implementation/models/{dataset_name}_LR_model.pkl"
            if not os.path.exists(model_path):
                self.get_logger().error(f"Model file not found at: {model_path}")
                return
            else:
                self.get_logger().info(f"Received model '{dataset_name}_LR_model.pkl'")
                self.send_acknowledgement() #informing training node that the model is retrieved

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Make predictions
            predictions = model.predict(X_test)

            # Evaluate predictions
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            # Save evaluation results to a text file
            model_path = f"/home/can_ozkan/ros2_ws/src/ros_ml_implementation/models/{dataset_name}_LR_model_results.txt"
            with open(model_path,"w") as file:
                file.write(f"---RESULTS OF {dataset_name}_LR_model: ---\n")
                file.write("\n---------------------")
                file.write(f"\n - Mean Absolute Error (MAE): {mae:.4f}")
                file.write(f"\n - R² Score: {r2:.4f}")
            
            self.get_logger().info(f"Evaluation results saved for '{dataset_name}' at {model_path}")

            # Create output directory for plots
            output_dir = f"/home/can_ozkan/ros2_ws/src/ros_ml_implementation/models/visuals"
            os.makedirs(output_dir, exist_ok=True)

            # Predicted vs Actual plot
            plt.figure(figsize=(6,6))
            plt.scatter(y_test, predictions, alpha=0.7)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title(f"{dataset_name} - Predicted vs Actual")
            plt.grid(True)
            plt.savefig(f"{output_dir}/{dataset_name}_predicted_vs_actual.png")
            plt.close()

            # Residual plot
            residuals = y_test - predictions
            plt.figure(figsize=(6,4))
            plt.scatter(predictions, residuals, alpha=0.7)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel("Predicted")
            plt.ylabel("Residuals")
            plt.title(f"{dataset_name} - Residual Plot")
            plt.grid(True)
            plt.savefig(f"{output_dir}/{dataset_name}_residual_plot.png")
            plt.close()

            # Histogram of residuals
            plt.figure(figsize=(6,4))
            plt.hist(residuals, bins=30, edgecolor='black')
            plt.title(f"{dataset_name} - Residual Histogram")
            plt.xlabel("Residual")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(f"{output_dir}/{dataset_name}_residual_hist.png")
            plt.close()

            # Optional: Line plot (if ordered data)
            plt.figure(figsize=(10,4))
            plt.plot(y_test, label='Actual')
            plt.plot(predictions, label='Predicted')
            plt.title(f"{dataset_name} - Actual vs Predicted (Ordered)")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{output_dir}/{dataset_name}_line_plot.png")
            plt.close()
            self.test_complete = True

        except Exception as e:
            self.get_logger().error(f"Error during model testing: {e}")


def main(args=None):
    """
    Main entry point for the ModelTester node.
    """
    rclpy.init(args=args)
    node = ModelTester()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
