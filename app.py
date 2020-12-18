from flask import Flask,request,render_template
from sklearn.ensemble import StackingClassifier
import numpy as np
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import io
import base64
from bs4 import BeautifulSoup
import numpy as np
import requests
import pandas as pd
from bs4 import BeautifulSoup
from sklearn import *

import string 
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize,pos_tag 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer




from io import BytesIO
import base64 
import os

app=Flask(__name__)

stack_bal=pickle.load(open('model.pkl','rb'))
trans=pickle.load(open('tfidf_v.pkl','rb'))

@app.route("/")

def home():
	return render_template('home.html')



@app.route('/sentiment',methods=['POST','GET'])
def sentiment():
    global model_predict
    model_predict=1
    if request.method=='POST':
        message=str(request.form['message'])
        data=prepro(message)
        data=[data]
        print(data)
        count_vect=CountVectorizer(ngram_range=(1,2))
        tfidf_transformer=TfidfTransformer()
        X_data=count_vect.fit_transform(data)        
        vector=tfidf_transformer.fit_transform(X_data).toarray()
        col_l=len(vector[0])
        siz=(1758-col_l)
        arr=np.zeros(siz)
        arr=arr.reshape(1,siz)
        vector=vector.reshape(1,col_l)
        vect_n=np.concatenate([vector,arr],axis=1)
        model_predict=stack_bal.predict(vect_n)
    return render_template('sentiment.html',prediction=model_predict)



def prepro(text1):
    lw=text_lowercase(text1)
    nu=remove_numbers(lw) 
    pu=remove_punctuation(nu)
    rw=remove_whitespace(pu)
    tok=remove_stopwords(rw)
    text2 = ' '.join(tok) 
    lemma_t=lemmatize_word(text2)
    text3 = ' '.join(lemma_t) 
    return text3

def text_lowercase(text): 
    return text.lower()  

def remove_numbers(text): 
    result = re.sub(r'\d+', '', text) 
    return result 

def remove_punctuation(text): 
    translator = str.maketrans('', '', string.punctuation) 
    return text.translate(translator) 

def remove_whitespace(text): 
    return  " ".join(text.split()) 

def remove_stopwords(text): 
    stop_words = set(stopwords.words("english")) 
    word_tokens = word_tokenize(text) 
    filtered_text = [word for word in word_tokens if word not in stop_words] 
    return filtered_text



def lemmatize_word(text):
	lemmatizer = WordNetLemmatizer()
	word_tokens = word_tokenize(text)
	lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens]
	return lemmas 

 
