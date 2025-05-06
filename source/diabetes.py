import mlflow
import mlflow.sklearn
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import numpy as np
import argparse
from typing import Dict, Any, Tuple, Optional, Union, List


def train_model(n_estimators: int, max_depth: int, learning_rate: float, subsample: float, colsample_bytree: float) -> Dict[str, Any]:
    """
    Train an XGBoost classifier on diabetes data with the given hyperparameters.
    
    Args:
        n_estimators: Number of gradient boosted trees
        max_depth: Maximum tree depth for base learners
        learning_rate: Boosting learning rate
        subsample: Subsample ratio of the training instances
        colsample_bytree: Subsample ratio of columns when constructing each tree
        
    Returns:
        Dictionary containing the trained model and evaluation metrics
    """
    mlflow.set_tracking_uri("file:mlruns")  # Fixed to use relative path
    
    results: Dict[str, Any] = {}
    
    with mlflow.start_run(nested=True):

        # Load and prepare dataset
        df = pd.read_csv('cleaned_data.csv')
        
        top_features: List[str] = ['HighBP', 'GenHlth', 'DiffWalk', 'BMI', 'HighChol', 
                            'HeartDiseaseorAttack', 'PhysHlth', 'Age', 'Stroke', 'Income']
        X = df[top_features]
        y = df['Diabetes_binary']

        # Apply SMOTE for balancing
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
        )

        # Train the model
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)

        # Evaluate
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report_dict = classification_report(y_test, predictions, output_dict=True)

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("subsample", subsample)
        mlflow.log_param("colsample_bytree", colsample_bytree)
        
        for key, value in report_dict.items():
            if isinstance(value, dict):
                for metric_name, metric_value in value.items():
                    if isinstance(metric_value, (float, int)):
                        mlflow.log_metric(f"{metric_name}_{key}", float(metric_value))
            elif isinstance(value, (float, int)):
                mlflow.log_metric(f"{key}", float(value))

        
        # Log model
        mlflow.sklearn.log_model(model, "xgboost_model")
        
        # Store results
        results["model"] = model
        results["accuracy"] = accuracy
        results["classification_report"] = report_dict
        
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model for diabetes prediction")
    parser.add_argument("--n_estimators", type=int, help="Number of gradient boosted trees")
    parser.add_argument("--max_depth", type=int, help="Maximum depth of trees")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--subsample", type=float, help="Subsample ratio")
    parser.add_argument("--colsample_bytree", type=float, help="Column subsample ratio")
    args = parser.parse_args()

    # Train model with command line arguments
    result = train_model(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree
    )
    
    print(f"Model training complete. Accuracy: {result['accuracy']:.4f}")