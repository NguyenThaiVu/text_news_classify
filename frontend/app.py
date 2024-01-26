import os
import streamlit as st
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from config import *
from utils import *


def predict_single_sentence(model, label_encoder, input_text):
    pred_label = model.predict([input_text])[0]
    pred_label = label_encoder.classes_[pred_label]

    pred_proba = model.predict_proba([input_text])[0]

    return pred_label, pred_proba



def show_word_cloud(input_text, width=800, height=400):
    wordcloud = WordCloud(width=width, height=height, background_color="white").generate(input_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)


def main():

    global model
    global label_encoder

    st.title("News Classification App")
    input_text = st.text_area("Enter Text:", "Type Here...", height=200)

    # Process text for prediction
    processed_text = preprocess_text(input_text) 

    if st.button("Predict"):
        
        pred_label, pred_proba = predict_single_sentence(model, label_encoder, processed_text)
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

        st.header("Word cloud:")
        show_word_cloud(input_text)


parent_directory = os.path.dirname(os.getcwd())

model = joblib.load(os.path.join(parent_directory, PATH_TRAINED_XGB))
label_encoder = joblib.load(os.path.join(parent_directory, LABEL_ENCODER_NAME))

if __name__ == "__main__":
    main()
