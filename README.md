# Satellite-Imagery-Based-Property-Valuation-Project
 Project Overview

Accurate house price prediction is a challenging real-world problem influenced not only by a property’s internal characteristics (such as size, number of rooms, and construction quality) but also by external environmental and neighborhood factors that are often difficult to quantify using structured data alone.

This project aims to build an end-to-end multimodal house price prediction system that combines:

Tabular housing data (property attributes, location, and engineered spatial features), and

Satellite imagery capturing neighborhood context such as surrounding infrastructure, greenery, road networks, and proximity to water or urban centers.

The primary objective is to predict house prices using both tabular and image-based features, while analyzing how much satellite images contribute beyond traditional tabular models. A strong tabular-only baseline is first established using XGBoost, followed by a multimodal learning pipeline where deep visual features extracted from satellite images are fused with tabular features for price prediction.

The project further demonstrates a realistic deployment scenario by performing inference using live satellite images fetched at prediction time, ensuring consistency between training and real-world usage. Through this approach, the project explores the strengths, limitations, and practical considerations of applying multimodal machine learning to real estate price estimation.
