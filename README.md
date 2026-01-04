# Satellite-Imagery-Based-Property-Valuation-Project
 Project Overview

Accurate house price prediction is a challenging real-world problem influenced not only by a property’s internal characteristics (such as size, number of rooms, and construction quality) but also by external environmental and neighborhood factors that are often difficult to quantify using structured data alone.

This project aims to build an end-to-end multimodal house price prediction system that combines:

Tabular housing data (property attributes, location, and engineered spatial features), and

Satellite imagery capturing neighborhood context such as surrounding infrastructure, greenery, road networks, and proximity to water or urban centers.

The primary objective is to predict house prices using both tabular and image-based features, while analyzing how much satellite images contribute beyond traditional tabular models. A strong tabular-only baseline is first established using XGBoost, followed by a multimodal learning pipeline where deep visual features extracted from satellite images are fused with tabular features for price prediction.

The project further demonstrates a realistic deployment scenario by performing inference using live satellite images fetched at prediction time, ensuring consistency between training and real-world usage. Through this approach, the project explores the strengths, limitations, and practical considerations of applying multimodal machine learning to real estate price estimation.

Project Pipeline (Execution Order)

1. Data Preprocessing (Tabular)
2. Satellite Image Fetching (Mapbox API)
3. Baseline Tabular Model Training
4. Image Feature Extraction (ResNet50)
5. Multimodal Model Training (Tabular + Image)
6. Live Multimodal Price Prediction (Test Data)

Environment & Installation

Python Version-Python 3.8+
Required Libraries:
Install all dependencies using-

pip install numpy pandas scikit-learn xgboost
pip install torch torchvision pillow
pip install category_encoders optuna joblib requests

📁 Required Input Files:

| File Name                        | Description                               |
| -------------------------------- | ----------------------------------------- |
| `tabular_Dataset.csv`            | Raw house data with prices & coordinates  |
| `preprocessed_data.csv`          | Cleaned & feature-engineered tabular data |
| `satellite_images/`              | Downloaded satellite images               |
| `image_download_summary.csv`     | Image-to-house ID mapping                 |
| `image_features_training.npy`    | Extracted image feature vectors           |
| `image_feature_ids_training.csv` | IDs corresponding to image features       |

Methodology

Tabular Data Preprocessing-

What this step does

Cleans raw data

Creates domain-aware features

Removes data leakage

Prepares data for ML models
