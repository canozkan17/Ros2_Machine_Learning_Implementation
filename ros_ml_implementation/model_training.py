#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import json
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import time
import os

class Trainer(Node):
    def __init__(self):
        super().__init__('node_model_trainer')
        self.subscription = self.create_subscription(
                                                        String,
                                                        'splitted_dataset',
                                                        self.listener_callback,
                                                        10
                                                    )
        self.publisher_training = self.create_publisher(Bool, "training_complete", 10)
        self.publisher_ack = self.create_publisher(String, "ack", 10)
        self.pose_subscriber_ack = self.create_subscription(String,"ack", self.recv_acknowledgement,10)
        self.model_trained = False
        self.current_dataset = ""
        self.acknowledged = False
        self.get_logger().info("Trainer node initialized and listening on 'splitted_dataset'")
        
    def send_acknowledgement(self): # to preprocessor node
        ack_msg = String()  
        ack_msg.data = json.dumps({
                                "sender": self.get_name(),
                                "data": True
                            })
        self.publisher_ack.publish(ack_msg)


    def recv_acknowledgement(self, msg: String): # receives from tester node
        parsed = json.loads(msg.data)
        if parsed["sender"] == "node_model_tester" and self.acknowledged != True:
            self.get_logger().info("Acklowledged = Trained model can be used")
            self.acknowledged = True
            
    
    def training_complete_publishing(self):
        out_msg = Bool()
        out_msg.data = True 
        self.publisher_training.publish(out_msg)
        

    def listener_callback(self, msg):
        if self.model_trained:
            parsed_msg = json.loads(msg.data)
            if self.current_dataset == parsed_msg["dataset_name"]:
                self.training_complete_publishing() 
                return
            self.model_trained = False
            self.acknowledged = False

        try:
            parsed_msg = json.loads(msg.data)
            dataset_name = parsed_msg["dataset_name"]
            self.current_dataset = dataset_name

            # Deserialize data
            X_train = pd.read_json(parsed_msg["X_train"], orient='split')
            y_train = pd.read_json(parsed_msg["y_train"], orient='split', typ='series')


            self.get_logger().info(f"Training model for dataset '{dataset_name}' on {X_train.shape[0]} samples")

            #timing the model training
            start_time = time.time()

            # Train Linear Regression model
            model = LinearRegression()
            model.fit(X_train, y_train)
            end_time = time.time()

            self.get_logger().info(f"Training completed in {end_time - start_time:.2f} seconds")

            # Save model
            model_path = f"/home/can_ozkan/ros2_ws/src/ros_ml_implementation/models/{dataset_name}_LR_model.pkl"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
                self.model_trained=True

            self.get_logger().info(f"Model training complete. Saved at '{model_path}'")

            # Publish training completion acknowledgment 
            self.send_acknowledgement()
            self.training_complete_publishing()
            

        except Exception as e:
            self.get_logger().error(f"Error during model training: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = Trainer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()