# ROS2 Linear Regression with Multiple Datasets

This repository implements a ROS2-based machine learning pipeline using Linear Regression on real-world datasets. It consists of modular ROS2 nodes for loading, preprocessing, training, and testing. The system is designed for data analysis in a distributed, scalable way via ROS middleware.

## 📁 Datasets Supported

- **new_height_weight.csv**: Human height vs. weight (10,000 rows)
- **HumanBrain_WeightandHead_size.csv**: Head size vs. brain weight (237 rows)
- **boston_housing.csv**: Boston housing features and prices (506 rows)

## 🧠 Node Architecture

| Node Name          | Description |
|-------------------|-------------|
| `node_loader`      | Loads CSV dataset, previews structure, and publishes to ROS topic. |
| `node_preprocessor`| Cleans and standardizes data, then splits into train/test sets. |
| `node_model_trainer`| Trains a linear regression model on incoming data. |
| `node_model_tester` | Tests the trained model and generates evaluation plots and metrics. |

## 🧪 Workflow

1. Launch all nodes in separate terminals.
2. Input the dataset name in the loader node when prompted.
3. The nodes will automatically handle data preprocessing, training, and testing.
4. Outputs include trained models, metrics, and plots.

## 🚀 Usage Instructions

```bash
# In different terminals:
ros2 run <your_package> dataset_loader
ros2 run <your_package> preprocessor
ros2 run <your_package> model_training
ros2 run <your_package> model_test
