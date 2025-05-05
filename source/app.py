from flask import Flask, jsonify, request
from flask_cors import CORS  # Import CORS
import pickle
import numpy as np
import mlflow.pyfunc

app = Flask(__name__)

# Enable CORS for all routes, allowing requests from your frontend origin
CORS(app, resources={r"/*": {"origins": "*"}})

# Load registered production models from MLflow Model Registry
diabetes_model = mlflow.pyfunc.load_model("models:/diabetes_production/3")
heart_model = mlflow.pyfunc.load_model("models:/heart_disease_production/1")

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Health Prediction API!"})

@app.route('/diabetes_test', methods=['POST'])
def diabetes_test():
    try:
        data = request.get_json()

        # Extract data from JSON
        HighBP = int(data['HighBP'])
        GenHlth = int(data['GenHlth'])
        DiffWalk = int(data['DiffWalk'])
        BMI = float(data['BMI'])
        HighChol = int(data['HighChol'])
        HeartDiseaseorAttack = int(data['HeartDiseaseorAttack'])
        PhysHlth = int(data['PhysHlth'])
        Age = int(data['Age'])
        Stroke = int(data['Stroke'])
        income = int(data['income'])

        input_data = np.array([[HighBP, GenHlth, DiffWalk, BMI, HighChol,
                                HeartDiseaseorAttack, PhysHlth, Age, Stroke, income]])

        prediction = diabetes_model.predict(input_data)
        result_msg = "You are likely to have diabetes." if prediction[0] == 1 else "You are unlikely to have diabetes."

        return jsonify({'prediction': int(prediction[0]), 'message': result_msg})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/heart_test', methods=['POST'])
def heart_test():
    try:
        data = request.get_json()

        # Extract data from JSON
        highbp = int(data['highbp'])
        diabetes = float(data['diabetes'])
        highchol = int(data['highchol'])
        stroke = int(data['stroke'])
        diffwalk = int(data['diffwalk'])
        genhlth = int(data['genhlth'])
        age = int(data['age'])
        physhlth = int(data['physhlth'])
        smoker = int(data['smoker'])
        income = int(data['income'])

        input_data = np.array([[highbp, diabetes, highchol, stroke, diffwalk,
                                genhlth, age, physhlth, smoker, income]])

        prediction = heart_model.predict(input_data)
        result_msg = "You are at risk for heart disease." if prediction[0] == 1 else "You are not at risk for heart disease."

        return jsonify({'prediction': int(prediction[0]), 'message': result_msg})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
