#!/usr/bin/python

%pip install neattext
from flask import Flask
from flask_restx import Api, Resource, fields
import sys
import os
import joblib
import pandas as pd

import neattext as nt
import neattext.functions as nfx
from sklearn.feature_extraction.text import TfidfVectorizer

def predict_genre(plot):
    
    tfidf = joblib.load('model_deployment/Xfeat.pkl') 
    clf = joblib.load('model_deployment/movie_genre_clf.pkl') 

    plot_ = pd.DataFrame([plot], columns=['plot'])
  
    # Create features
    plot_['plot'].apply(lambda x:nt.TextFrame(x).noise_scan())
    plot_['plot'].apply(lambda x:nt.TextExtractor(x).extract_stopwords())
    plot_['plot'].apply(nfx.remove_stopwords)
    plot_['plot'].apply(nfx.remove_stopwords)

    Xfeatures = tfidf.transform(plot_).toarray()
 
    # Make prediction
    p1 = clf.predict_proba(Xfeatures)

    cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']
    
    genre = pd.DataFrame(p1, columns=cols)
    return genre.transpose().rename(columns = {0:'Proba'})


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add an plot')
        
    else:

        plot = sys.argv[1]

        p1 = predict_genre(plot)
        
        print(plot)
        print('Probability of genres: ', p1)
        
