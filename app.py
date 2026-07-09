import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('movie_revenue_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))
st.title("🎬 Movie Revenue Predictor")
st.write("Predict movie revenue using a Random Forest Machine Learning Model")
budget = st.number_input(
    "Budget (USD)",
    min_value=1000,
    value=10000000
)

popularity = st.number_input(
    "Popularity",
    min_value=0.0,
    value=10.0
)

vote_average = st.slider(
    "Vote Average",
    0.0,
    10.0,
    5.0
)

vote_count = st.number_input(
    "Vote Count",
    min_value=0,
    value=100
)

runtime = st.number_input(
    "Runtime (Minutes)",
    min_value=30,
    value=120
)

release_year = st.number_input(
    "Release Year",
    min_value=1900,
    max_value=2030,
    value=2020
)

release_month = st.selectbox(
    "Release Month",
    list(range(1, 13))
)

genre = st.selectbox(
    "Primary Genre",
    [
        'Action',
        'Adventure',
        'Animation',
        'Comedy',
        'Crime',
        'Documentary',
        'Drama',
        'Family',
        'Fantasy',
        'History',
        'Horror',
        'Music',
        'Mystery',
        'Romance',
        'Science Fiction',
        'Thriller',
        'Unknown',
        'War',
        'Western'
    ]
)
if st.button("Predict Revenue"):

    # Create input dictionary
    input_data = {
        'log_budget': np.log1p(budget),
        'popularity': popularity,
        'vote_average': vote_average,
        'vote_count': vote_count,
        'runtime': runtime,
        'release_year': release_year,
        'release_month': release_month
    }

    # Create all genre columns with 0
    for col in feature_columns:
        if col.startswith('genre_'):
            input_data[col] = 0

    # Set selected genre to 1
    selected_genre_col = f'genre_{genre}'

    if selected_genre_col in input_data:
        input_data[selected_genre_col] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Numerical features used during training
    numeric_features = [
        'log_budget',
        'popularity',
        'vote_average',
        'vote_count',
        'runtime',
        'release_year',
        'release_month'
    ]

    # Scale numerical features
    input_df[numeric_features] = scaler.transform(
        input_df[numeric_features]
    )

    # Match training feature order
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Predict log revenue
    predicted_log_revenue = model.predict(input_df)[0]

    # Convert back to actual revenue
    predicted_revenue = np.expm1(
        predicted_log_revenue
    )

    # Display result
    st.success(
        f"🎬 Predicted Revenue: ${predicted_revenue:,.2f}"
)