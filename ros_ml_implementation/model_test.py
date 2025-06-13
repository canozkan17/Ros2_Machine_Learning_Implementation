#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import pandas as pd
import pickle
import os
from sklearn.metrics import mean_absolute_error, r2_score


class ModelTester(Node):
    def __init__(self):
        super().__init__('node_model_tester')
        self.subscription = self.create_subscription(
            String,
            'splitted_dataset',
            self.listener_callback,
            10
        )
        self.get_logger().info("Tester node initialized and listening on 'splitted_dataset'")

    def listener_callback(self, msg: String):
        try:
            parsed_msg = json.loads(msg.data)
            dataset_name = parsed_msg["dataset_name"]

            self.get_logger().info(f"Received test data for dataset '{dataset_name}'")

            # Deserialize test data
            X_test = pd.read_json(parsed_msg["X_test"], orient='split')
            y_test = pd.read_json(parsed_msg["y_test"], orient='split', typ='series')

            # Load trained model
            model_path = f"/home/can_ozkan/ros2_ws/src/ros_ml_implementation/models/{dataset_name}_LR_model.pkl"
            if not os.path.exists(model_path):
                self.get_logger().error(f"Model file not found at: {model_path}")
                return

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Make predictions
            predictions = model.predict(X_test)

            # Evaluate predictions
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            self.get_logger().info(f"Evaluation results for '{dataset_name}':")
            self.get_logger().info(f" - Mean Absolute Error (MAE): {mae:.4f}")
            self.get_logger().info(f" - R² Score: {r2:.4f}")

        except Exception as e:
            self.get_logger().error(f"Error during model testing: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ModelTester()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
