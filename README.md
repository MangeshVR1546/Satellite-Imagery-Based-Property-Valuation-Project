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


## ▶️ Execution Steps

---

### Step 1: Tabular Data Preprocessing

**Script:** Data preprocessing code  

#### Input
- `tabular_Dataset.csv`

#### Output
- `preprocessed_data.csv`
- `zipcode_target_encoder.pkl`

#### Purpose
- Cleans raw data  
- Creates spatial, temporal, and engineered features  

---

### Step 2: Satellite Image Download

**Script:** Image fetching code  

#### Input
- `tabular_Dataset.csv`
- Mapbox API Token

#### Output
- `satellite_images/`
- `image_download_summary.csv`

#### Purpose
- Downloads satellite images using latitude & longitude  

---

### Step 3: Baseline Tabular Model Training

**Script:** Baseline tabular model  

#### Input
- `preprocessed_data.csv`

#### Output
- Console metrics (R², RMSE)

#### Purpose
- Establishes tabular-only performance baseline  

---

### Step 4: Image Feature Extraction

**Script:** Image feature extraction  

#### Input
- `satellite_images/`
- `image_download_summary.csv`

#### Output
- `image_features_training.npy`
- `image_feature_ids_training.csv`

#### Purpose
- Extracts visual features from satellite images (ResNet50)  
- Uses random 20% sample for efficiency  

---

### Step 5: Multimodal Model Training

**Script:** Multimodal training  

#### Input
- `preprocessed_data.csv`
- `image_features_training.npy`
- `image_feature_ids_training.csv`

#### Output
- `multimodal_pipeline.pkl`
- Console metrics (R², RMSE)

#### Purpose
- Trains combined tabular + image model  

---

### Step 6: Test Set Prediction (Live Satellite)

**Script:** Live multimodal prediction  

#### Input
- Test CSV  
- `multimodal_pipeline.pkl`  
- Mapbox API Token  

#### Output
- `24117072_final.csv`

#### Purpose
- Performs real-time satellite-based inference  

---


##  Methodology and Design Rationale

---

### 1️⃣ Tabular Data Preprocessing

#### What this step does
This step prepares the raw housing dataset for machine learning by cleaning inconsistencies, engineering meaningful features, and removing potential data leakage. The output is a structured dataset that can be directly consumed by regression models.

#### Key Feature Engineering
- **Temporal features** (`sale_year`, `property_age`) to capture time-dependent price behavior  
- **Target transformation** (`log_price`) to stabilize variance and improve regression performance  
- **Renovation indicator** to capture value uplift due to upgrades  
- **Geospatial features** using Haversine distance to:
  - City center  
  - Water bodies  
  - Major tech hubs  
- **Location clustering** using KMeans to group spatially similar properties  
- **Target encoding** for high-cardinality categorical feature `zipcode`

#### Why this approach was used
House prices exhibit strong **non-linear relationships** and are heavily influenced by **location and proximity effects**. Distance-based features encode spatial economics more effectively than raw latitude and longitude. Target encoding preserves price-related information from categorical variables without causing dimensional explosion, making it well-suited for tree-based models.

#### Output
- `preprocessed_data.csv`  
- `zipcode_target_encoder.pkl`

---

### 2️⃣ Satellite Image Fetching (Mapbox API)

#### What this step does
This step downloads **high-resolution satellite images** for each house using geographic coordinates. These images provide visual context about the surrounding environment that is unavailable in tabular data.

#### Why satellite images?
Satellite imagery captures:
- Neighborhood quality  
- Urban density  
- Road connectivity  
- Green cover and proximity to water  

Such contextual signals often influence property prices but are difficult to encode numerically.

#### Design choices
- **Zoom level 18** to capture neighborhood-scale details  
- **512×512 resolution** for sufficient spatial context  
- **Multi-threaded downloading** to reduce execution time  
- **Rate limiting** to comply with API constraints  

#### Output
- `satellite_images/`  
- `image_download_summary.csv`

---

### 3️⃣ Baseline Model (Tabular Only)

#### Model used
- **XGBoost Regressor**

#### Why XGBoost?
XGBoost is a state-of-the-art algorithm for structured data. It efficiently models non-linear relationships, captures feature interactions, and is robust to noise and scaling issues.

#### Target strategy
- Predict `log(price)` during training  
- Convert predictions back to price scale during evaluation  

#### Performance
- **R² ≈ 0.898**  
- **RMSE ≈ $113,000**

#### Why the baseline is important
The tabular-only model serves as a **strong reference point**, allowing a fair and meaningful comparison when image features are introduced. It helps quantify the **true contribution of satellite imagery** rather than relying on intuition.

---

### 4️⃣ Image Feature Extraction (ResNet50)

#### What this step does
This step extracts **deep visual features** from satellite images using a pretrained ResNet50 model. Each image is converted into a **512-dimensional embedding** representing high-level visual patterns.

#### Why a pretrained CNN?
Training a CNN from scratch is infeasible due to dataset size and computational constraints. Models pretrained on ImageNet learn general-purpose visual representations that transfer well to satellite imagery. Using frozen features also reduces overfitting risk.

#### Role of ResNet50 in This Project

ResNet50 is used as a **deep feature extractor** to convert raw satellite images into meaningful numerical representations that can be consumed by traditional machine learning models.

ResNet50 is a 50-layer deep convolutional neural network that uses **residual (skip) connections**, enabling stable training of very deep architectures. When pretrained on ImageNet, it learns rich hierarchical visual features such as edges, textures, shapes, and spatial patterns, which transfer well to satellite imagery.

In this project:
- The **pretrained ResNet50** model is loaded from `torchvision`
- The final classification layer is removed
- The network is used in **inference-only mode**
- Each satellite image is converted into a **512-dimensional feature vector**

These features capture:
- Urban density patterns  
- Road and building layouts  
- Presence of greenery or water bodies  
- Neighborhood-level visual structure  

Using ResNet50 avoids the need to train a deep CNN from scratch, which would require significantly more labeled data and computational resources.

---

#### Why ResNet50 Was Chosen

- **Proven architecture** with strong generalization ability  
- **Pretrained weights** available for immediate use  
- **Balanced depth** (deep enough to capture complex patterns, but not overly heavy)  
- Widely used in **computer vision and satellite image analysis** tasks  

Alternative architectures (e.g., training a custom CNN) were avoided due to limited data and higher overfitting risk.

---

#### Integration with the Multimodal Pipeline

The ResNet50 image embeddings are:
1. Scaled using `RobustScaler`
2. Reduced in dimensionality using **PCA (512 → 64)**
3. Concatenated with scaled tabular features
4. Passed to an **XGBoost regressor** for final price prediction

This design allows deep visual information to be incorporated into a **non-deep, interpretable multimodal regression framework**.

#### Why sample only 20% of images?
Satellite image processing is computationally expensive. Sampling a subset:
- Keeps training feasible  
- Reflects realistic multimodal constraints  
- Allows experimentation without excessive compute cost  

#### Output
- `image_features_training.npy`  
- `image_feature_ids_training.csv`

---

### 5️⃣ Multimodal Model Training

#### What this step does
This step combines **tabular features and image features** into a unified learning pipeline. Both modalities are scaled, image features are reduced using PCA, and the final model is trained using XGBoost.

#### Key techniques
- **RobustScaler** to handle outliers  
- **PCA (512 → 64)** to reduce noise and redundancy in image embeddings  
- **Optuna** for automated hyperparameter optimization  
- **House ID alignment** to ensure correct tabular–image pairing  

#### Why PCA on image features?
Raw CNN embeddings are high-dimensional and noisy. PCA preserves dominant visual patterns while preventing image features from overwhelming tabular signals.

#### Model architecture

[ Tabular Features ] ─┐
├─> XGBoost Regressor

[ Image PCA Features ]┘


#### Output
- `multimodal_pipeline.pkl`

---

### 6️⃣ Live Multimodal Prediction (Test Data)

#### What this step does
This step performs **end-to-end inference** by:
- Fetching satellite images live at prediction time  
- Extracting image features on-the-fly  
- Applying the trained multimodal pipeline  
- Generating final house price predictions  

#### Why live satellite inference?
This design simulates **real-world deployment**, where images may not be pre-stored. It demonstrates production readiness and ensures consistency between training and inference pipelines.

#### Output
- `24117072_final.csv`

---
