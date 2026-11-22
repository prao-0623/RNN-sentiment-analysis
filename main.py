from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()} 

#Load pretrained model 
model = load_model(Path(__file__).with_name('simple_rnn_imdb_v2.keras'), compile=False)

#helper function
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

#function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=model.input_shape[-1])
    return padded_review

#prediction function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

import streamlit as st

st.title('TMDB movie sentiment analysis')

#user input
user_input = st.text_area('Movie review')
if st.button('Clasify'):
    preprocessed_input=preprocess_text(user_input)

    #make prediction
    prediction=model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction > 0.5 else 'Negative'

    #display result
    st.write(f'Sentiment:{sentiment}')
    st.write(f'Prediction score:{prediction[0][0]}')
else:
    st.write('Please enter a movie review')