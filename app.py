import os
import sys
import streamlit as st
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import requests
import json

from config import *
from utils import *




def show_word_cloud(input_text, width=800, height=400):
    wordcloud = WordCloud(width=width, height=height, background_color="white").generate(input_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)


def main():
    global label_encoder
    
    st.title("News Classification App")
    input_text = st.text_area("Enter Text:", "Type Here...", height=200)

    # Process text for prediction
    processed_text = preprocess_text(input_text) 

    if st.button("Predict"):
        
        # Send input sentence to backend
        data = {'processed_text': processed_text}
        response = requests.post(f"{BASE_URL}/predict", json=data)

        # Check if the request was successful
        if response.status_code == 200:
            
            # Get output response
            result = response.json()
            pred_label = result['pred_label']
            pred_proba = result['pred_proba']

            # Show predicted output
            st.success(f"Predicted label: {pred_label}")

            st.header("Predicted probabilities")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Categories")
                for item in list(label_encoder.classes_):
                    st.write(item)

            with col2:
                st.subheader("Probabilities:")
                for proba in list(pred_proba):
                    proba = round(proba * 100, 2)
                    st.write(f"{proba} %")

        else:
            st.error("Failed to get prediction result")


        st.header("Word cloud:")
        show_word_cloud(input_text)


# Load the model
parent_directory = os.getcwd()
label_encoder = joblib.load(os.path.join(parent_directory, LABEL_ENCODER_NAME))

if __name__ == "__main__":
    main()
