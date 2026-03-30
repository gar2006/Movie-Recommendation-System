# Movie-Recommendation-System
Movie Recommender System

A content-based Movie Recommendation System that suggests similar movies based on genres, keywords, and plot descriptions using Natural Language Processing (NLP) techniques. The application features an interactive web interface built with Streamlit and integrates the OMDb API to display real-time movie details, posters, and summaries.

Live Demo:
https://movie-recommendation-system-aeewnvmjzizmxcnzfmxukr.streamlit.app/

Features:
Content-based movie recommendations using TF-IDF and Cosine Similarity
Interactive and user-friendly web interface built with Streamlit
Real-time movie information fetched using the OMDb API
Displays movie posters, release year, genre, and plot summaries
Efficient text preprocessing using NLTK
Optimized performance using Streamlit caching
Responsive UI with custom CSS styling

How It Works:
Movie data is loaded from a dataset containing:
Genres
Keywords
Overview (plot description)
Text data is cleaned using:
Tokenization
Stopword removal
Lowercasing

The cleaned text is converted into numerical vectors using:
TF-IDF Vectorization

Similarity between movies is calculated using:
Cosine Similarity

When a user selects a movie, the system recommends the most similar movies based on their content.

Technologies Used
Python
Streamlit
Pandas
Scikit-learn
NLTK
Requests
TF-IDF Vectorizer
Cosine Similarity
OMDb API
Custom CS
