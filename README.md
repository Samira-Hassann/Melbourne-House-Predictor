# 🏠 Melbourne Housing Price Predictor App

An end-to-end Machine Learning web application built with **Streamlit** and **CatBoost** to predict real estate prices in the Melbourne Housing Market. The project includes extensive data cleaning, anomaly detection, feature engineering, and hyperparameter tuning using Optuna.

🔗 **Live Demo:** [Melbourne House Predictor App](https://melbourne-house-predictor.streamlit.app/)  
📓 **Kaggle Notebook:** [Melbourne Housing Market Analysis & Price Prediction](https://www.kaggle.com/code/samoura/melbourne-housing-market-analysis-price-predition)

---

## 📌 Project Overview

The Melbourne housing dataset presents several real-world data quality challenges (e.g., impossible year built values, inconsistent room vs. bedroom counts, extreme outliers). This project handles these data anomalies, preprocesses features through a pipeline, trains an optimized **CatBoost Regressor**, and deploys the model via an interactive Streamlit web dashboard.

---

## 🛠️ Key Features & Data Processing

- **Data Sanitization & Outlier Filtering:**
  - Removed extreme price anomalies (e.g., multi-million dollar inconsistencies).
  - Cleaned non-logical bedroom/room ratios ($\text{Bedroom2} > \text{Rooms}$) and unrealistic car space counts.
  - Reset invalid zero-landsize entries for houses and townhouses to `NaN`.
  - Filtered non-logical land sizes ($> 5000\text{ m}^2$).
- **Feature Engineering & Refinement:**
  - Categorical casting for identifiers like `Postcode`.
  - Extracted temporal features (`Year`, `Month`, `Day_of_week`) from transaction dates.
- **Machine Learning Pipeline:**
  - Trained and tuned top-performing algorithms using **Optuna**.
  - Final model selection: **CatBoost Regressor** bundled into a serialized pipeline (`catboost_house_price_pipeline.pkl`).

---

## 📁 Repository Structure

```text
├── Melbourne_House_Predictor_app.py   # Main Streamlit web application
├── catboost_house_price_pipeline.pkl  # Pre-trained ML pipeline (CatBoost + Transformers)
├── requirements.txt                   # Python dependencies
├── runtime.txt                        # Specified Python environment runtime
└── README.md                          # Project documentation
