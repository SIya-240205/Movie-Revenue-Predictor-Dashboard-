# Movie-Revenue-Predictor-Dashboard

An end-to-end Machine Learning project that predicts the expected revenue of a movie using historical movie metadata. The project covers the complete machine learning pipeline, including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, hyperparameter tuning, and deployment using Streamlit.

## Project Overview

The objective of this project is to estimate a movie's revenue before its release based on features such as budget, popularity, ratings, vote count, runtime, release year, release month, and genre.

The project demonstrates the complete workflow of building and deploying a regression-based machine learning model using real-world data.

## Features

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Log Transformation
- One-Hot Encoding
- Feature Scaling using StandardScaler
- Linear Regression Model
- Random Forest Regressor
- Hyperparameter Tuning using GridSearchCV
- Feature Importance Analysis
- Interactive Streamlit Dashboard

## Dataset

**Dataset:** TMDB Movie Metadata Dataset

The dataset contains movie-related information including:
- Budget
- Revenue
- Genres
- Popularity
- Vote Average
- Vote Count
- Runtime
- Release Date
- Production Companies
- Director
- Cast

After preprocessing, the final dataset consisted of **4,604 movies**.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Pickle

## Exploratory Data Analysis

The following analyses were performed:
- Revenue Distribution
- Budget vs Revenue
- Movie Rating vs Revenue
- Popularity vs Revenue
- Correlation Heatmap
- Top 10 Genres by Average Revenue
- Top 10 Most Frequent Genres

## Feature Engineering

The following feature engineering techniques were applied:
- Log Transformation of Revenue
- Log Transformation of Budget
- Extraction of Release Year
- Extraction of Release Month
- Primary Genre Extraction
- One-Hot Encoding of Genres
- Standard Scaling of Numerical Features

## Machine Learning Models

Two regression models were trained and evaluated:

### 1. Linear Regression

Used as the baseline regression model.

### 2. Random Forest Regressor

Selected as the final model because it achieved better performance across all evaluation metrics.

## Model Performance

| Model | R² Score | MAE | RMSE |
|--------|---------:|----:|-----:|
| Linear Regression | 0.496 | 0.942 | 1.354 |
| Random Forest | **0.544** | **0.866** | **1.288** |

## 🔧 Hyperparameter Tuning

Hyperparameter tuning was performed using **GridSearchCV**.

**Parameters Tuned**

- `n_estimators`
- `max_depth`

**Best Parameters**

```python
{
    'n_estimators': 300,
    'max_depth': 20
}
```

Although GridSearchCV identified the optimal parameter combination, the original Random Forest model performed slightly better on the unseen test data and was therefore retained for deployment.

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard that allows users to:
- Enter movie details
- Predict movie revenue instantly
- View prediction results in an easy-to-use interface

## Project Structure

```
Movie-Revenue-Predictor/
│
├── Movie_Revenue_Predictor.ipynb
├── app.py
├── movie_revenue_model.pkl
├── scaler.pkl
├── README.md
├── requirements.txt
└── screenshots/
```
## How to Run
1. Open the notebook and run all cells.
2. Run:
streamlit run app.py

## Key Findings
- Budget is the strongest predictor of movie revenue.
- Adventure and Animation movies generate the highest average revenue.
- Random Forest outperformed Linear Regression with an R² score of 0.5444.
- A Streamlit dashboard was developed for interactive revenue prediction.

## 🔮 Future Improvements

- XGBoost Regressor
- Marketing Budget Analysis
- Actor & Director Popularity
- Social Media Sentiment Analysis
- Movie Description NLP
- Larger Movie Dataset

---

## Author

**Siya Anand**

Integrated M.Tech in Data Science  
VIT Bhopal University

---

## If you found this project useful, consider giving it a star!
