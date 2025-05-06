#!/usr/bin/env python3
"""
Test script for the diabetes prediction endpoint of the Health Prediction API.
This script sends test requests to verify the API's functionality and response handling.
"""

import requests
import json
import logging
from typing import Dict, Any, List, Optional
import sys
import time
from requests.exceptions import RequestException

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('diabetes_test_results.log')
    ]
)
logger = logging.getLogger('diabetes_api_test')

# API endpoint
API_URL = "https://healthcare-depi-project-production.up.railway.app"
DIABETES_ENDPOINT = f"{API_URL}/diabetes_test"

def test_api_connection() -> bool:
    """Test basic connectivity to the API root endpoint."""
    try:
        logger.info("Testing API connectivity...")
        response = requests.get(API_URL)
        if response.status_code == 200:
            logger.info("API connection successful!")
            return True
        else:
            logger.error(f"API connection failed with status code: {response.status_code}")
            return False
    except RequestException as e:
        logger.error(f"API connection error: {str(e)}")
        return False

def send_diabetes_test_request(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Send a test request to the diabetes prediction endpoint.
    
    Args:
        payload: Dictionary containing the test data
        
    Returns:
        API response as a dictionary, or None if the request failed
    """
    try:
        logger.info(f"Sending request to diabetes endpoint with payload: {payload}")
        headers = {"Content-Type": "application/json"}
        response = requests.post(DIABETES_ENDPOINT, data=json.dumps(payload), headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Received successful response: {result}")
            return result
        else:
            logger.error(f"Request failed with status code {response.status_code}: {response.text}")
            return None
    except RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return None

def run_test_cases() -> None:
    """Run a series of test cases for the diabetes endpoint."""
    
    # Test case 1: Likely diabetes case
    test_case_1 = {
        "HighBP": 1,
        "GenHlth": 4,
        "DiffWalk": 1,
        "BMI": 32.5,
        "HighChol": 1,
        "HeartDiseaseorAttack": 1,
        "PhysHlth": 15,
        "Age": 65,
        "Stroke": 1,
        "income": 2
    }
    
    # Test case 2: Unlikely diabetes case
    test_case_2 = {
        "HighBP": 0,
        "GenHlth": 1,
        "DiffWalk": 0,
        "BMI": 22.5,
        "HighChol": 0,
        "HeartDiseaseorAttack": 0,
        "PhysHlth": 0,
        "Age": 30,
        "Stroke": 0,
        "income": 7
    }
    
    # Test case 3: Missing field
    test_case_3 = {
        "HighBP": 1,
        "GenHlth": 2,
        "DiffWalk": 0,
        "BMI": 27.5,
        # Missing HighChol
        "HeartDiseaseorAttack": 0,
        "PhysHlth": 3,
        "Age": 50,
        "Stroke": 0,
        "income": 4
    }
    
    # Test case 4: Invalid value type
    test_case_4 = {
        "HighBP": 1,
        "GenHlth": 2,
        "DiffWalk": 0,
        "BMI": "not_a_number",  # Invalid BMI value
        "HighChol": 1,
        "HeartDiseaseorAttack": 0,
        "PhysHlth": 3,
        "Age": 50,
        "Stroke": 0,
        "income": 4
    }
    
    # Run the tests
    test_cases = [
        ("Test Case 1 (Likely Diabetes)", test_case_1),
        ("Test Case 2 (Unlikely Diabetes)", test_case_2),
        ("Test Case 3 (Missing Field)", test_case_3),
        ("Test Case 4 (Invalid Value)", test_case_4)
    ]
    
    for name, test_case in test_cases:
        logger.info(f"Running {name}...")
        result = send_diabetes_test_request(test_case)
        
        # Add a pause between requests to avoid overwhelming the server
        time.sleep(1)
    
    logger.info("All test cases completed")

def main() -> None:
    """Main function to run the test script."""
    logger.info("Starting Diabetes Endpoint Test")
    
    if not test_api_connection():
        logger.error("Could not connect to API. Make sure the server is running.")
        sys.exit(1)
    
    run_test_cases()
    
    logger.info("Diabetes Endpoint Test completed")

if __name__ == "__main__":
    main()