import json, re, os, requests
import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e6e1;
}

.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 2rem;
}
.hero-badge {
    display: inline-block;
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #c9a84c;
    border: 1px solid rgba(201,168,76,0.3);
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 0.75rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #f0ece3;
    margin: 0 0 0.3rem;
    letter-spacing: -0.02em;
}
.hero p { font-size: 15px; color: #6a6865; font-weight: 300; margin: 0; }

.section-meta { margin-bottom: 1.2rem; }
.section-meta small {
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #5a5855;
}
.section-meta h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    color: #f0ece3;
    margin: 2px 0 0;
}

.movie-card {
    background: #12101a;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    overflow: hidden;
    transition: border-color 0.2s;
}
.movie-card:hover { border-color: rgba(201,168,76,0.35); }
.movie-card img { width: 100%; display: block; }
.movie-card-body { padding: 12px 14px 14px; }
.movie-card-title {
    font-size: 14px;
    font-weight: 500;
    color: #f0ece3;
    margin: 0 0 5px;
    line-height: 1.3;
}
.movie-card-plot {
    font-size: 12px;
    color: #6a6865;
    line-height: 1.5;
    margin: 0;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.match-badge {
    display: inline-block;
    font-size: 11px;
    color: #c9a84c;
    background: rgba(201,168,76,0.1);
    border: 1px solid rgba(201,168,76,0.25);
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 6px;
}

div[data-testid="stSelectbox"] > div:first-child {
    background: #16141e !important;
    border-color: rgba(255,255,255,0.1) !important;
    color: #e8e6e1 !important;
    border-radius: 8px !important;
}
.stButton > button {
    background: #c9a84c !important;
    color: #0a0a0f !important;
    border: none !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.6rem !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button:hover { background: #d4b560 !important; }
div[data-testid="stAlert"] {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.1) !important;
}
</style>

<div class="hero">
  <div class="hero-badge">AI-Powered Discovery</div>
  <h1>🎬 Movie Recommender</h1>
  <p>Select a film you love — we'll find what to watch next.</p>
</div>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/gar2006/Movie-Recommendation-System/refs/heads/main/movies.csv"
    df = pd.read_csv(url)
    df = df[['genres', 'keywords', 'overview', 'title']].dropna().reset_index(drop=True)
    df['combined'] = df['genres'] + " " + df['keywords'] + " " + df['overview']

    for pkg, path in [('punkt', 'tokenizers/punkt'), ('punkt_tab', 'tokenizers/punkt_tab'),
                      ('stopwords', 'corpora/stopwords')]:
        try: nltk.data.find(path)
        except LookupError: nltk.download(pkg, quiet=True)

    stop_words = set(stopwords.words('english'))

    def clean(text):
        text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
        return " ".join(w for w in word_tokenize(text) if w not in stop_words)

    df['cleaned'] = df['combined'].apply(clean)
    return df

@st.cache_resource
def build_similarity(df):
    vec = TfidfVectorizer(max_features=5000)
    mat = vec.fit_transform(df['cleaned'])
    return cosine_similarity(mat)

df = load_data()
similarity = build_similarity(df)

# ── OMDB ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "config.json")) as f:
    OMDB_API_KEY = json.load(f)["OMDB_API_KEY"]

@st.cache_data(show_spinner=False)
def get_movie_details(title):
    res = requests.get("http://www.omdbapi.com/",
                       params={"t": title, "plot": "short", "apikey": OMDB_API_KEY}).json()
    if res.get("Response") == "True":
        return res.get("Plot", ""), res.get("Poster", ""), res.get("Year", ""), res.get("Genre", "")
    return "", "", "", ""

def recommend(movie_name, top_n=9):
    idx = df[df['title'].str.lower() == movie_name.lower()].index
    if len(idx) == 0:
        return None
    scores = sorted(enumerate(similarity[idx[0]]), key=lambda x: x[1], reverse=True)[1:top_n+1]
    results = df.iloc[[i for i, _ in scores]][['title']].copy()
    results['score'] = [round(s * 100) for _, s in scores]
    return results

# ── UI ────────────────────────────────────────────────────────────────────────
movie_list = sorted(df['title'].unique())
col_sel, col_btn = st.columns([4, 1])
with col_sel:
    selected = st.selectbox("Choose a film you love", movie_list, label_visibility="collapsed")
with col_btn:
    go = st.button("Find Similar", use_container_width=True)

if go:
    with st.spinner("Finding your next watch..."):
        recs = recommend(selected)

    if recs is None or recs.empty:
        st.warning("No recommendations found for that title.")
    else:
        st.markdown(f"""
        <div class="section-meta">
          <small>Because you liked</small>
          <h2>{selected}</h2>
        </div>""", unsafe_allow_html=True)

        cols = st.columns(3)
        for i, (_, row) in enumerate(recs.iterrows()):
            plot, poster, year, genre = get_movie_details(row['title'])
            with cols[i % 3]:
                card_html = f'<div class="movie-card">'
                if poster and poster != "N/A":
                    card_html += f'<img src="{poster}" alt="{row["title"]}">'
                card_html += f"""
                <div class="movie-card-body">
                  <div class="match-badge">{row['score']}% match</div>
                  <div class="movie-card-title">{row['title']}{' · ' + year if year else ''}</div>
                  {f'<p class="movie-card-plot">{plot}</p>' if plot else ''}
                </div></div>"""
                st.markdown(card_html, unsafe_allow_html=True)
                st.markdown("")
