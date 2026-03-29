import pandas as pd
import numpy as np
from wordcloud import WordCloud


df=pd.read_csv(r"C:\Users\garim\Downloads\movies.csv")

required=['genres','keywords','overview','title']

df=df[required]


df=df.dropna().reset_index(drop=True)
df['combined']=df['genres']+" "+df['keywords']+" "+df['overview']

data=df[['title','combined']]
text=" ".join(df['combined'])
wordcloud=WordCloud(width=800,height=400,background_color='white').generate(text)

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
plt.figure(figsize=(12,6))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.title('Keyword')
plt.show()


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words=set(stopwords.words('english'))


import re
def text_cleaning(text):
    #removing special characters
    text=re.sub(r"[^a-zA-z\s]","",text)
    #convert it into lower case
    text=text.lower()
    #tokenzise and remove stopwords
    tokens=word_tokenize(text)
    tokens=[word for word in tokens if word not in stop_words]
    return " ".join(tokens)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#create vectors for the above movies
vector=TfidfVectorizer(max_features=5000)
matrix=vector.fit_transform(data['cleaned'])
similarity=cosine_similarity(matrix,matrix)


def recommend(movie_name,sim=similarity,df=data,top_n=10):
    idx=df[df['title'].str.lower()==movie_name.lower()].index
    if len(idx)==0:
        return "Movie not found!"
    idx=idx[0]

    score=list(enumerate(sim[idx]))
    score=sorted(score,key=lambda x:x[1],reverse=True)
    score=score[1:top_n+1]

    movie_indices=[i[0] for i in score]

    return df[['title']].iloc[movie_indices]

