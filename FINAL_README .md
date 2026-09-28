# ProphetEstate: End-to-End AI & Machine Learning Valuation Engine 🏠

A complete machine learning project built for **Module 6: Final AI & ML Project**.

## Project Summary
ProphetEstate is a full-lifecycle Machine Learning system designed to evaluate residential properties. The pipeline automates data ingestion, cleaning, one-hot categorical encoding, algorithm benchmarking, model persistence, and deployment-ready inference.

## Key Technical Features
- **Data Engineering:** Automated missing value handling, column schema enforcement, and one-hot encoding across categorical dimensions (`Location`, `Condition`, `Garage`).
- **Benchmarking Suite:** Evaluates multiple baseline and ensemble regression algorithms:
  - Linear Regression
  - Ridge Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- **Evaluation Metrics:** Quantified using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and $R^2$ Score.
- **Model Serialization:** Exports the top-performing estimator and column signatures using `joblib` for rapid inference without retraining.
- **Interactive Interface:** CLI client for instant valuation estimations.

## Project Structure
```text
├── House Price Prediction Dataset.csv  # 2,000-record property dataset
├── model_pipeline.py                  # Data processing & training pipeline
├── main.py                            # Interactive CLI valuation interface
├── requirements.txt                   # Dependency specifications
└── README.md                          # Project documentation