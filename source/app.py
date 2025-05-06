from flask import Flask, jsonify, request, Response
from flask_cors import CORS  # Import CORS
import pickle
import numpy as np
import mlflow.pyfunc
from typing import Dict, Any, List, Union, Optional, Tuple
import logging
from logging.handlers import RotatingFileHandler
import os
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create logger
logger = logging.getLogger('health_prediction_api')

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# Create file handler
file_handler = RotatingFileHandler('health_api.log', maxBytes=10485760, backupCount=5)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

app = Flask(__name__)

# Enable CORS for all routes, allowing requests from your frontend origin
CORS(app, resources={r"/*": {"origins": "*"}})

logger.info("Starting Health Prediction API")

# Load registered production models from MLflow Model Registry
try:
    logger.info("Loading diabetes model from MLflow Model Registry")
    diabetes_model = mlflow.pyfunc.load_model("models:/diabetes_production/1")
    logger.info("Successfully loaded diabetes model")
    
    logger.info("Loading heart disease model from MLflow Model Registry")
    heart_model = mlflow.pyfunc.load_model("models:/heart_disease_production/1")
    logger.info("Successfully loaded heart disease model")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    logger.error(traceback.format_exc())
    raise

@app.route('/')
def index() -> Response:
    """Return welcome message for API root endpoint."""
    logger.info("Index endpoint accessed")
    return jsonify({"message": "Welcome to the Health Prediction API!"})

@app.route('/diabetes_test', methods=['POST'])
def diabetes_test() -> Tuple[Response, int]:
    """
    Endpoint for diabetes prediction.
    
    Expected JSON payload:
    {
        "HighBP": int,
        "GenHlth": int,
        "DiffWalk": int,
        "BMI": float,
        "HighChol": int,
        "HeartDiseaseorAttack": int,
        "PhysHlth": int,
        "Age": int,
        "Stroke": int,
        "income": int
    }
    
    Returns:
        Prediction result and message
    """
    logger.info("Diabetes test endpoint accessed")
    try:
        data: Dict[str, Any] = request.get_json()
        logger.debug(f"Received data: {data}")

        # Extract data from JSON
        HighBP: int = int(data['HighBP'])
        GenHlth: int = int(data['GenHlth'])
        DiffWalk: int = int(data['DiffWalk'])
        BMI: float = float(data['BMI'])
        HighChol: int = int(data['HighChol'])
        HeartDiseaseorAttack: int = int(data['HeartDiseaseorAttack'])
        PhysHlth: int = int(data['PhysHlth'])
        Age: int = int(data['Age'])
        Stroke: int = int(data['Stroke'])
        income: int = int(data['income'])

        input_data: np.ndarray = np.array([[HighBP, GenHlth, DiffWalk, BMI, HighChol,
                                HeartDiseaseorAttack, PhysHlth, Age, Stroke, income]])
        
        logger.info("Making diabetes prediction")
        prediction: np.ndarray = diabetes_model.predict(input_data)
        prediction_value: int = int(prediction[0])
        
        result_msg: str = "You are likely to have diabetes." if prediction_value == 1 else "You are unlikely to have diabetes."
        logger.info(f"Diabetes prediction result: {prediction_value}, message: {result_msg}")

        return jsonify({'prediction': prediction_value, 'message': result_msg}), 200

    except KeyError as e:
        error_msg = f"Missing required field: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 400
    except ValueError as e:
        error_msg = f"Invalid value for field: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 400
    except Exception as e:
        logger.error(f"Error in diabetes_test: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 400

@app.route('/heart_test', methods=['POST'])
def heart_test() -> Tuple[Response, int]:
    """
    Endpoint for heart disease prediction.
    
    Expected JSON payload:
    {
        "highbp": int,
        "diabetes": float,
        "highchol": int,
        "stroke": int,
        "diffwalk": int,
        "genhlth": int,
        "age": int,
        "physhlth": int,
        "smoker": int,
        "income": int
    }
    
    Returns:
        Prediction result and message
    """
    logger.info("Heart test endpoint accessed")
    try:
        data: Dict[str, Any] = request.get_json()
        logger.debug(f"Received data: {data}")

        # Extract data from JSON
        highbp: int = int(data['highbp'])
        diabetes: float = float(data['diabetes'])
        highchol: int = int(data['highchol'])
        stroke: int = int(data['stroke'])
        diffwalk: int = int(data['diffwalk'])
        genhlth: int = int(data['genhlth'])
        age: int = int(data['age'])
        physhlth: int = int(data['physhlth'])
        smoker: int = int(data['smoker'])
        income: int = int(data['income'])

        input_data: np.ndarray = np.array([[highbp, diabetes, highchol, stroke, diffwalk,
                                genhlth, age, physhlth, smoker, income]])
        
        logger.info("Making heart disease prediction")
        prediction: np.ndarray = heart_model.predict(input_data)
        prediction_value: int = int(prediction[0])
        
        result_msg: str = "You are at risk for heart disease." if prediction_value == 1 else "You are not at risk for heart disease."
        logger.info(f"Heart disease prediction result: {prediction_value}, message: {result_msg}")

        return jsonify({'prediction': prediction_value, 'message': result_msg}), 200

    except KeyError as e:
        error_msg = f"Missing required field: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 400
    except ValueError as e:
        error_msg = f"Invalid value for field: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 400
    except Exception as e:
        logger.error(f"Error in heart_test: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    logger.info("Starting Flask server")
    app.run(debug=True)