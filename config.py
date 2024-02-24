import os

# Data
LABEL_ENCODER_NAME = os.path.join('models', 'label_encoder.joblib')

# Model
NUM_CLASSES = 5
PATH_TRAINED_XGB = os.path.join("models", "trained_xgb.joblib")

# Backend server
HOST = os.environ.get("SERVING_HOST", "127.0.0.1")
PORT = os.environ.get("SERVING_PORT", "5000")
BASE_URL = os.environ.get("SERVING_URL", "")
if BASE_URL == "":
    BASE_URL = f"http://{HOST}:{PORT}"