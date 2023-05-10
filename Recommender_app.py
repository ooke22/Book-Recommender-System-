import pickle
import streamlit as st
import numpy as np
import pandas as pd

st.header('Book Recommender System')

@st.cache_resource
def load_model(model_name):
    book_pivtable = pickle.load(model_name)
    return book_pivtable

model = pickle.load(open('C:/Books_Recommender_System/book_recommender.pkl', 'rb'))
book_titles = pickle.load(open('C:/Books_Recommender_System/book_titles.pkl', 'rb'))
final_ratings = pickle.load(open('C:/Books_Recommender_System/final.pkl', 'rb'))
book_pivtable = load_model(open('C:/Books_Recommender_System/book_pivot.pkl', 'rb'))
#book_pivtable = pickle.load(open('C:/Books_Recommender_System/book_pivot.pkl', 'rb'))

def fetch_poster(sugesstion):
    book_name = []
    ids_index = []
    poster_url = []

    for book_id in sugesstion:
        book_name.append(book_pivtable.index[book_id])

    for name in book_name[0]:
        ids = np.where(final_ratings['Book-Title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final_ratings.iloc[idx]['Image-URL-L']
        poster_url.append(url)

    return poster_url

def book_recommender(book_titles):
    book_list = []
    book_id = np.where(book_pivtable.index == book_titles)[0][0]
    distance, suggestion = model.kneighbors(book_pivtable.iloc[book_id,:].values.reshape(1, -1), n_neighbors=6)

    poster_url = fetch_poster(suggestion)

    for i in range(len(suggestion)):
        books = book_pivtable.index[suggestion[i]]
        for j in books:
            book_list.append(j)

    return book_list, poster_url

user_selected_books = st.selectbox(
    "Search books",
    book_titles
)

if st.button('Show Recommended Books'):
    book_recommendation, poster_url = book_recommender(user_selected_books)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(book_recommendation[1])
        st.image(poster_url[1])

    with col2:
        st.text(book_recommendation[2])
        st.image(poster_url[2])

    with col3:
        st.text(book_recommendation[3])
        st.image(poster_url[3])

    with col4:
        st.text(book_recommendation[4])
        st.image(poster_url[4])

    with col5:
        st.text(book_recommendation[5])
        st.image(poster_url[5])