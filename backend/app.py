import os
import sys
import joblib
from flask import Flask, request, jsonify

from backend_config import *
from utils import *


app = Flask(__name__)

# Load the model
model = joblib.load(PATH_TRAINED_XGB)
LABEL_ENCODER = {0:'business', 1:'entertainment', 2:'politics', 3:'sport', 4:'tech'}



def predict_single_sentence(input_text):
    global model

    pred_label = model.predict([input_text])[0]
    pred_label = LABEL_ENCODER[pred_label]

    pred_proba = model.predict_proba([input_text])[0]

    return pred_label, pred_proba


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """

    # Get RAW input data
    data = request.get_json()
    input_text = data['input_text']

    # Process text for prediction
    processed_text = preprocess_text(input_text) 

    pred_label, pred_proba = predict_single_sentence(processed_text)

    output = {'pred_label': pred_label, 'pred_proba': pred_proba.tolist()}
    return jsonify(output)


if __name__ == '__main__':
    app.run(port=5000, debug=True)