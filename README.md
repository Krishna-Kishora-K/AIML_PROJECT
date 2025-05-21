# Machine Learning Projects Collection

This repository contains two exciting machine learning projects:

1. **Movie Recommendation System**  
2. **Credit Card Fraud Detection**

---

## Project 1: Movie Recommendation System

### Project Objective

Suggest movies based on user preferences using machine learning techniques.

### Tools & Technologies Used

- Python  
- Pandas  
- Scikit-learn  
- Streamlit  
- MovieLens Dataset (`movies.csv`)

### How It Works

1. **Dataset Used**  
   - Uses MovieLens `movies.csv` dataset containing movie titles and genres.  
2. **Data Preprocessing**  
   - Cleaned genre strings by replacing `|` with spaces.  
   - Applied TF-IDF Vectorizer to convert genre text into numerical vectors.  
3. **Model Logic**  
   - Calculated cosine similarity between movies based on genre vectors.  
   - Found top 5 similar movies for any selected movie title.  
4. **User Interface (Streamlit App)**  
   - Dropdown menu to select a movie.  
   - “Recommend” button shows the top 5 recommended movies.

# Project 2: Credit Card Fraud Detection

## Project Overview

Credit card fraud is a serious problem with significant financial implications. Due to the imbalanced nature of fraud data (very few fraudulent transactions compared to normal ones), detecting fraud accurately requires careful data processing and a robust classifier.

This project:

- Preprocesses and balances the dataset using under-sampling.
- Trains an **XGBoost** classifier on selected features.
- Evaluates the model using multiple performance metrics.
- Deploys a **Streamlit** web app for real-time fraud prediction.

## How It Works

### Data Preprocessing

- The dataset is highly imbalanced (less than 0.2% fraud cases).
- Applied **Random Under-Sampling** to balance the classes.
- Split the dataset into training and testing sets.

### Model Training

- An **XGBoost Classifier** was trained on the resampled data.
- Model performance evaluated using Accuracy, Precision, Recall, F1-score, and ROC-AUC.

### Model Deployment

- The trained model saved as a `.pkl` file using `joblib`.
- Streamlit app allows users to input transaction data and get real-time fraud predictions.


```bash
pip install -r requirements.txt
streamlit run fraud_detection_app.py

