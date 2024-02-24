import os
import numpy as np
import joblib
import requests
from flask import Flask, request, jsonify

from config import *

app = Flask(__name__)

# Load the model
parent_directory = os.getcwd()
model = joblib.load(os.path.join(parent_directory, PATH_TRAINED_XGB))
label_encoder = joblib.load(os.path.join(parent_directory, LABEL_ENCODER_NAME))


def predict_single_sentence(input_text):
    global model
    global label_encoder

    pred_label = model.predict([input_text])[0]
    pred_label = label_encoder.classes_[pred_label]

    pred_proba = model.predict_proba([input_text])[0]

    return pred_label, pred_proba


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """

    # Get input data
    data = request.get_json()
    processed_text = data['processed_text']

    pred_label, pred_proba = predict_single_sentence(processed_text)

    output = {'pred_label': pred_label, 'pred_proba': pred_proba.tolist()}
    return jsonify(output)


if __name__ == '__main__':
    app.run(port=5000, debug=True)