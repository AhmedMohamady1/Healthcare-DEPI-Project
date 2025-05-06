from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import numpy as np
import mlflow.pyfunc
import logging
from logging.handlers import RotatingFileHandler
import traceback
from typing import Dict, Any, Tuple

# --- Logging Setup ---
def setup_logger(name: str = 'health_prediction_api') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    file_handler = RotatingFileHandler('health_api.log', maxBytes=10 * 1024 * 1024, backupCount=5)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Load Models from MLflow ---
def load_model_from_registry(model_name: str, stage: str = "1"):
    try:
        logger.info(f"Loading model: {model_name}")
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")
        logger.info(f"{model_name} model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading {model_name} model: {e}")
        logger.error(traceback.format_exc())
        raise

diabetes_model = load_model_from_registry("diabetes_production")
heart_model = load_model_from_registry("heart_disease_production")

# --- Error Handler Decorator ---
def safe_route(handler):
    def wrapper(*args, **kwargs):
        try:
            return handler(*args, **kwargs)
        except KeyError as e:
            logger.error(f"Missing field: {e}")
            return jsonify({'error': f'Missing field: {str(e)}'}), 400
        except ValueError as e:
            logger.error(f"Invalid value: {e}")
            return jsonify({'error': f'Invalid value: {str(e)}'}), 400
        except Exception as e:
            logger.error(f"Unhandled error: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    wrapper.__name__ = handler.__name__
    return wrapper

# --- Utility Functions ---
def predict(model, input_data: Dict[str, Any], expected_order: list, messages: Tuple[str, str]) -> Tuple[Response, int]:
    values = [float(input_data[field]) for field in expected_order]
    input_array = np.array([values])
    logger.info(f"Predicting with input: {values}")
    prediction = int(model.predict(input_array)[0])
    message = messages[1] if prediction == 1 else messages[0]
    logger.info(f"Prediction: {prediction}, Message: {message}")
    return jsonify({'prediction': prediction, 'message': message}), 200

# --- Routes ---
@app.route('/')
def index() -> Response:
    logger.info("Root endpoint accessed")
    return jsonify({"message": "Welcome to the Health Prediction API!"})

@app.route('/diabetes_test', methods=['POST'])
@safe_route
def diabetes_test() -> Tuple[Response, int]:
    data = request.get_json()
    fields = ["HighBP", "GenHlth", "DiffWalk", "BMI", "HighChol",
              "HeartDiseaseorAttack", "PhysHlth", "Age", "Stroke", "income"]
    return predict(
        model=diabetes_model,
        input_data=data,
        expected_order=fields,
        messages=("You are unlikely to have diabetes.", "You are likely to have diabetes.")
    )

@app.route('/heart_test', methods=['POST'])
@safe_route
def heart_test() -> Tuple[Response, int]:
    data = request.get_json()
    fields = ["highbp", "diabetes", "highchol", "stroke", "diffwalk",
              "genhlth", "age", "physhlth", "smoker", "income"]
    return predict(
        model=heart_model,
        input_data=data,
        expected_order=fields,
        messages=("You are not at risk for heart disease.", "You are at risk for heart disease.")
    )

# --- Start Server ---
if __name__ == '__main__':
    logger.info("Starting Flask server")
    app.run(debug=True)
