# Credit Card Fraud Detection

This project focuses on building a machine learning model that detects fraudulent credit card transactions. Using a publicly available dataset, the model is trained to identify suspicious activity based on transaction features. A Streamlit-based user interface allows real-time predictions for new transactions.

## Project Overview

Credit card fraud is a serious problem with significant financial implications. Due to the imbalanced nature of fraud data (very few fraudulent transactions compared to normal ones), detecting fraud accurately requires careful data processing and a robust classifier.

This project:
- Preprocesses and balances the dataset using under-sampling.
- Trains an **XGBoost** classifier on selected features.
- Evaluates the model using multiple performance metrics.
- Deploys a **Streamlit** web app for easy interaction with the trained model.

---

## How It Works

1. **Data Preprocessing**:
   - The dataset is highly imbalanced (less than 0.2% fraud cases).
   - We apply **Random Under-Sampling** to balance the classes.
   - The dataset is split into training and testing sets.

2. **Model Training**:
   - An **XGBoost Classifier** is trained on the resampled data.
   - Model performance is evaluated using Accuracy, Precision, Recall, F1-score, and ROC-AUC.

3. **Model Deployment**:
   - The trained model is saved as a `.pkl` file using `joblib`.
   - A **Streamlit** web app allows users to input transaction data and get real-time fraud predictions.

---


