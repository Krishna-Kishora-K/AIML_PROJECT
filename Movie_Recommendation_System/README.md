# Movie Recommendation System

This project is a simple **Content-Based Movie Recommender System** that suggests movies based on genre similarity using **TF-IDF vectorization** and **cosine similarity**. It includes a clean and interactive UI built using **Streamlit**.

---

## Project Objective

> Suggest movies based on user preferences using machine learning techniques.

---

## Tools & Technologies Used

- **Python**  
- **Pandas**  
- **Scikit-learn**  
- **Streamlit**  
- **MovieLens Dataset (movies.csv)**

---

## How It Works

1. **Dataset Used**  
   - The project uses the MovieLens `movies.csv` dataset which contains movie titles and genre information.

2. **Data Preprocessing**  
   - Cleaned genre strings (replaced `|` with space).  
   - Applied **TF-IDF Vectorizer** to convert genre text into vectors.

3. **Model Logic**  
   - Calculated **cosine similarity** between movie vectors.  
   - Used similarity scores to find the top 5 similar movies for any selected title.

4. **User Interface (Streamlit App)**  
   - Dropdown menu to select a movie.  
   - A “Recommend” button shows the top 5 recommended movies.

---


