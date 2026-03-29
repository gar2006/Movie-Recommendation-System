import json
import streamlit as st
import pandas as pd
import re
import requests
import nltk
import matplotlib.pyplot as plt
import os

from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

url="https://raw.githubusercontent.com/gar2006/Movie-Recommendation-System/refs/heads/main/movies.csv"
df=pd.read_csv(url)

required = ['genres', 'keywords', 'overview', 'title']
df = df[required].dropna().reset_index(drop=True)

df['combined'] = df['genres'] + " " + df['keywords'] + " " + df['overview']
data = df[['title', 'combined']]
 
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
     

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
 
def text_cleaning(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

data['cleaned'] = data['combined'].apply(text_cleaning)
 
vector = TfidfVectorizer(max_features=5000)
matrix = vector.fit_transform(data['cleaned'])
similarity = cosine_similarity(matrix)

 
def recommend(movie_name, top_n=10):
    idx = data[data['title'].str.lower() == movie_name.lower()].index
    if len(idx) == 0:
        return None

    idx = idx[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:top_n + 1]

    movie_indices = [i[0] for i in scores]
    return data.iloc[movie_indices][['title']]
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to config.json inside repository
config_path = os.path.join(BASE_DIR, "config.json")

# Load config
with open(config_path, "r") as f:
    config = json.load(f)

OMDB_API_KEY = config["OMDB_API_KEY"]


def get_movie_details(title):
    url = "http://www.omdbapi.com/"
    params = {"t": title, "plot": "full", "apikey": OMDB_API_KEY}
    res = requests.get(url, params=params).json()

    if res.get("Response") == "True":
        return res.get("Plot", "N/A"), res.get("Poster", "N/A")

    return "N/A", "N/A"
 
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

st.title("🎬 Movie Recommender")

movie_list = sorted(df['title'].unique())
selected_movie = st.selectbox("🎬 Select a movie:", movie_list)

if st.button("🚀 Recommend Similar Movies"):
    with st.spinner("Finding similar movies..."):
        recommendations = recommend(selected_movie)

        if recommendations is None or recommendations.empty:
            st.warning("Sorry, no recommendations found.")
        else:
            st.success("Top similar movies:")
            columns=3
            cols=st.columns(columns)
            for idx, (_, row) in enumerate(recommendations.iterrows()):
                col=cols[idx%columns]

                movie_title = row['title']
                plot, poster = get_movie_details(movie_title)

                with col:
                    if poster != "N/A":
                        st.image(poster, use_container_width=True)
                    st.markdown(f"### {movie_title}")
 

text = " ".join(data['cleaned'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

 
