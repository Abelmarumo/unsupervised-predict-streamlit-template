"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.feature_extraction.text import TfidfVectorizer
# Custom Libraries
from utils.data_loader import load_movie_titles
#from recommenders.collaborative_based import collab_model
#from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')
df = pd.read_csv('resources/data/df_clean_gen.csv')

tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2),
                     min_df=0, stop_words='english')

tf_gen_matrix = tf.fit_transform(df['genre'].iloc[:40000])

@st.cache
def sim_matrix(tf_gen_matrix):
    return cosine_similarity(tf_gen_matrix, tf_gen_matrix)

cosine_sim_authTags =sim_matrix(tf_gen_matrix)
titles = df['title']
ind_titles = pd.Series(df.index,index=titles)

def content_generate_top_N_recommendations(movie_list, N=10):
    m_idx1 = ind_titles[movie_list[0]]
    m_idx2 = ind_titles[movie_list[1]]
    m_idx3 = ind_titles[movie_list[2]]
    
    s1=list(enumerate(cosine_sim_authTags[m_idx1]))
    s2=list(enumerate(cosine_sim_authTags[m_idx2]))
    s3=list(enumerate(cosine_sim_authTags[m_idx3]))
    
    s1 = sorted(s1, key=lambda x: x[1], reverse=True)[1:N]
    s2 = sorted(s2, key=lambda x: x[1], reverse=True)[1:N]
    s3 = sorted(s3, key=lambda x: x[1], reverse=True)[1:N]
    
    s_all = sorted(s1+s2+s3,key=lambda x: x[1], reverse=True)
    s_10 = s_all[1:N+1] 
    movie_indices = [i[0] for i in s_10] 
    
    return titles.iloc[movie_indices]
# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Exploritory Analysis"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_generate_top_N_recommendations(fav_movies)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations.values):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        #if sys == 'Collaborative Based Filtering':
            #if st.button("Recommend"):
                #try:
                    #with st.spinner('Crunching the numbers...'):
                        #top_recommendations = collab_model(movie_list=fav_movies,
                                                           #top_n=10)
                    #st.title("We think you'll like:")
                    #for i,j in enumerate(top_recommendations):
                        #st.subheader(str(i+1)+'. '+j)
                #except:
                    #st.error("Oops! Looks like this algorithm does't work.\
                              #We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Exploritory Analysis":
        st.title("Solution Overview")
        st.write("Describe your winning approach on this page")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
