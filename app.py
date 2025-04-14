import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Page Configuration
st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🍿", layout="centered")

# Header
#st.title("🎬 Movie Recommendation System")
#st.markdown("Welcome to the intelligent movie recommender! Just type a movie title below and get similar movies you might love. 💖")

# Load movie data
movies = pd.read_csv('movie_data.csv')

# Fill NaNs
movies['soup'] = movies['soup'].fillna('')

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['soup'])

# Fit model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(tfidf_matrix)

# Index mapping
title_mapping = movies['title'].str.lower().reset_index()
indices = pd.Series(title_mapping['index'].values, index=title_mapping['title']).to_dict()

# Recommendation function
def get_recommendations(title, n=10):
    title = title.lower()
    if title not in indices:
        return f"❌ Movie '{title}' not found in the database. Please check the spelling."

    idx = indices[title]
    distances, indices_found = model.kneighbors(tfidf_matrix[idx], n_neighbors=n + 1)
    recommendations = movies.iloc[indices_found[0][1:]][['title', 'release_date', 'vote_average']]
    recommendations = recommendations.reset_index(drop=True)
    recommendations.index += 1
    return recommendations

# Sidebar with app description
st.sidebar.title("About This App")
st.sidebar.write("""
    This Movie Recommendation App uses **Content-Based Filtering** to suggest movies based on your search query.
    
    1. **Input the name of a movie** you're interested in.
    2. **Get personalized recommendations** based on movie details like genre, overview, and cast.
""")


# UI Input
#movie_name = st.text_input("Enter a movie title:")
# UI setup
st.title("🎬 Movie Recommendation App")
st.markdown("Get similar movies based on your favorite one!")

# Dropdown for movie selection
movie_titles = movies['title'].sort_values().tolist()
movie_name = st.selectbox("🎥 Select a movie from the list or type to search:", movie_titles)


# Recommend Button
if st.button("Recommend"):
    if movie_name:
        with st.spinner('Fetching recommendations...'):
            results = get_recommendations(movie_name)
            if isinstance(results, str):
                st.error(results)
            else:
                st.success("Here are your recommendations! 🎉")
                st.dataframe(results)
    else:
        st.warning("Please enter a movie name first!")
