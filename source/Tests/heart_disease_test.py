#!/usr/bin/env python3
"""
Test script for the heart disease prediction endpoint of the Health Prediction API.
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
        logging.FileHandler('heart_disease_test_results.log')
    ]
)
logger = logging.getLogger('heart_disease_api_test')

# API endpoint
API_URL = "https://healthcare-depi-project-production.up.railway.app"
HEART_ENDPOINT = f"{API_URL}/heart_test"

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

def send_heart_test_request(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Send a test request to the heart disease prediction endpoint.
    
    Args:
        payload: Dictionary containing the test data
        
    Returns:
        API response as a dictionary, or None if the request failed
    """
    try:
        logger.info(f"Sending request to heart disease endpoint with payload: {payload}")
        headers = {"Content-Type": "application/json"}
        response = requests.post(HEART_ENDPOINT, data=json.dumps(payload), headers=headers)
        
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
    """Run a series of test cases for the heart disease endpoint."""
    
    # Test case 1: High risk for heart disease
    test_case_1 = {
        "highbp": 1,
        "diabetes": 1.0,
        "highchol": 1,
        "stroke": 1,
        "diffwalk": 1,
        "genhlth": 5,
        "age": 70,
        "physhlth": 15,
        "smoker": 1,
        "income": 1
    }
    
    # Test case 2: Low risk for heart disease
    test_case_2 = {
        "highbp": 0,
        "diabetes": 0.0,
        "highchol": 0,
        "stroke": 0,
        "diffwalk": 0,
        "genhlth": 1,
        "age": 25,
        "physhlth": 0,
        "smoker": 0,
        "income": 8
    }
    
    # Test case 3: Missing field
    test_case_3 = {
        "highbp": 1,
        "diabetes": 0.0,
        "highchol": 1,
        # Missing stroke field
        "diffwalk": 0,
        "genhlth": 3,
        "age": 45,
        "physhlth": 5,
        "smoker": 1,
        "income": 5
    }
    
    # Test case 4: Invalid value type
    test_case_4 = {
        "highbp": 1,
        "diabetes": "not_a_number",  # Invalid diabetes value
        "highchol": 1,
        "stroke": 0,
        "diffwalk": 0,
        "genhlth": 3,
        "age": 45,
        "physhlth": 5,
        "smoker": 1,
        "income": 5
    }
    
    # Run the tests
    test_cases = [
        ("Test Case 1 (High Risk)", test_case_1),
        ("Test Case 2 (Low Risk)", test_case_2),
        ("Test Case 3 (Missing Field)", test_case_3),
        ("Test Case 4 (Invalid Value)", test_case_4)
    ]
    
    for name, test_case in test_cases:
        logger.info(f"Running {name}...")
        result = send_heart_test_request(test_case)
        
        # Add a pause between requests to avoid overwhelming the server
        time.sleep(1)
    
    logger.info("All test cases completed")

def main() -> None:
    """Main function to run the test script."""
    logger.info("Starting Heart Disease Endpoint Test")
    
    if not test_api_connection():
        logger.error("Could not connect to API. Make sure the server is running.")
        sys.exit(1)
    
    run_test_cases()
    
    logger.info("Heart Disease Endpoint Test completed")

if __name__ == "__main__":
    main()