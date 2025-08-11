# API Usage Examples

This document provides comprehensive examples of how to use the Property Friends ML API.

## Quick Start Examples

### Python Requests

```python
import requests
import json

# API configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "demo-api-key-for-testing"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Single property prediction
def predict_property_price(property_features):
    url = f"{API_BASE_URL}/api/v1/predict"
    
    payload = {
        "features": property_features,
        "request_id": "example-001"  # Optional
    }
    
    response = requests.post(url, json=payload, headers=HEADERS)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
        return None

# Example property
casa_vitacura = {
    "type": "casa",
    "sector": "vitacura", 
    "net_usable_area": 140.0,
    "net_area": 170.0,
    "n_rooms": 4,
    "n_bathroom": 3,
    "latitude": -33.40123,
    "longitude": -70.58056
}

# Make prediction
result = predict_property_price(casa_vitacura)
if result:
    print(f"Predicted price: ${result['prediction']:,.2f} CLP")
    print(f"Model version: {result['model_version']}")
    print(f"Request ID: {result['request_id']}")
```

### cURL Examples

```bash
# Health check
curl -X GET "http://localhost:8000/api/v1/health"

# Single prediction
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Authorization: Bearer demo-api-key-for-testing" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "type": "casa",
      "sector": "vitacura",
      "net_usable_area": 140.0,
      "net_area": 170.0,
      "n_rooms": 4,
      "n_bathroom": 3,
      "latitude": -33.40123,
      "longitude": -70.58056
    }
  }'

# Batch prediction
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Authorization: Bearer demo-api-key-for-testing" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "type": "casa",
      "sector": "vitacura",
      "net_usable_area": 140.0,
      "net_area": 170.0,
      "n_rooms": 4,
      "n_bathroom": 3,
      "latitude": -33.40123,
      "longitude": -70.58056
    },
    {
      "type": "departamento",
      "sector": "las condes",
      "net_usable_area": 80.0,
      "net_area": 95.0,
      "n_rooms": 2,
      "n_bathroom": 2,
      "latitude": -33.41135,
      "longitude": -70.56977
    }
  ]'

# Model information
curl -X GET "http://localhost:8000/api/v1/model/info" \
  -H "Authorization: Bearer demo-api-key-for-testing"
```

## Advanced Usage

### Batch Processing Script

```python
import pandas as pd
import requests
from typing import List, Dict
import time

class PropertyPricePredictor:
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def predict_single(self, features: Dict) -> Dict:
        """Predict price for a single property."""
        url = f"{self.api_url}/api/v1/predict"
        response = requests.post(url, json={"features": features}, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    def predict_batch(self, features_list: List[Dict], batch_size: int = 10) -> List[Dict]:
        """Predict prices for multiple properties with batching."""
        results = []
        
        for i in range(0, len(features_list), batch_size):
            batch = features_list[i:i + batch_size]
            
            url = f"{self.api_url}/api/v1/predict/batch"
            response = requests.post(url, json=batch, headers=self.headers)
            
            if response.status_code == 200:
                results.extend(response.json())
            else:
                print(f"Batch {i//batch_size + 1} failed: {response.status_code}")
                # Add error placeholders
                for _ in batch:
                    results.append({"prediction": None, "error": "API Error"})
            
            # Rate limiting
            time.sleep(0.1)
        
        return results
    
    def process_csv(self, input_file: str, output_file: str):
        """Process a CSV file of properties and save predictions."""
        # Read CSV
        df = pd.read_csv(input_file)
        
        # Convert to list of dictionaries
        features_list = df.to_dict('records')
        
        # Make predictions
        predictions = self.predict_batch(features_list)
        
        # Add predictions to dataframe
        df['predicted_price'] = [p.get('prediction') for p in predictions]
        df['prediction_error'] = [p.get('error') for p in predictions]
        
        # Save results
        df.to_csv(output_file, index=False)
        print(f"Processed {len(df)} properties, saved to {output_file}")

# Usage
predictor = PropertyPricePredictor(
    api_url="http://localhost:8000",
    api_key="demo-api-key-for-testing"
)

# Process a CSV file
# predictor.process_csv("properties_to_evaluate.csv", "properties_with_predictions.csv")
```

### Error Handling

```python
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def create_session_with_retries():
    """Create a requests session with retry logic."""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        method_whitelist=["HEAD", "GET", "POST"],
        backoff_factor=1
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def robust_prediction(features: Dict, api_url: str, api_key: str):
    """Make a prediction with robust error handling."""
    session = create_session_with_retries()
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        response = session.post(
            f"{api_url}/api/v1/predict",
            json={"features": features},
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            return {"error": "Authentication failed - check API key"}
        elif response.status_code == 422:
            return {"error": "Invalid input data", "details": response.json()}
        else:
            return {"error": f"API error: {response.status_code}"}
            
    except requests.exceptions.Timeout:
        return {"error": "Request timeout"}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection error - check if API is running"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
```

## Integration Examples

### Django Integration

```python
# In your Django views.py
import requests
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def predict_property_price(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        
        # Call ML API
        ml_api_url = settings.ML_API_URL
        ml_api_key = settings.ML_API_KEY
        
        headers = {"Authorization": f"Bearer {ml_api_key}"}
        response = requests.post(
            f"{ml_api_url}/api/v1/predict",
            json={"features": data},
            headers=headers
        )
        
        if response.status_code == 200:
            return JsonResponse(response.json())
        else:
            return JsonResponse(
                {'error': 'ML API error', 'details': response.text},
                status=500
            )
            
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
```

### Flask Integration

```python
from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

ML_API_URL = os.getenv('ML_API_URL', 'http://localhost:8000')
ML_API_KEY = os.getenv('ML_API_KEY', 'demo-api-key-for-testing')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        property_data = request.json
        
        headers = {"Authorization": f"Bearer {ML_API_KEY}"}
        response = requests.post(
            f"{ML_API_URL}/api/v1/predict",
            json={"features": property_data},
            headers=headers
        )
        
        return jsonify(response.json()), response.status_code
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

## Monitoring and Logging

### Checking API Health

```python
def check_api_health(api_url: str) -> Dict:
    """Check if the API is healthy and return status."""
    try:
        response = requests.get(f"{api_url}/api/v1/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            return {
                "status": "healthy" if health_data.get("status") == "healthy" else "unhealthy",
                "model_loaded": health_data.get("model_loaded", False),
                "model_trained": health_data.get("model_trained", False),
                "api_responsive": True
            }
        else:
            return {
                "status": "unhealthy",
                "api_responsive": False,
                "error": f"HTTP {response.status_code}"
            }
            
    except Exception as e:
        return {
            "status": "unhealthy", 
            "api_responsive": False,
            "error": str(e)
        }

# Usage
health = check_api_health("http://localhost:8000")
print(f"API Status: {health['status']}")
```

### Performance Monitoring

```python
import time
from typing import List
import statistics

def benchmark_api_performance(api_url: str, api_key: str, num_requests: int = 100) -> Dict:
    """Benchmark API performance with multiple requests."""
    headers = {"Authorization": f"Bearer {api_key}"}
    
    sample_features = {
        "type": "casa",
        "sector": "vitacura",
        "net_usable_area": 140.0,
        "net_area": 170.0,
        "n_rooms": 4,
        "n_bathroom": 3,
        "latitude": -33.40123,
        "longitude": -70.58056
    }
    
    response_times = []
    errors = 0
    
    for i in range(num_requests):
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{api_url}/api/v1/predict",
                json={"features": sample_features},
                headers=headers,
                timeout=30
            )
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # ms
            
            if response.status_code == 200:
                response_times.append(response_time)
            else:
                errors += 1
                
        except Exception:
            errors += 1
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.01)
    
    if response_times:
        return {
            "total_requests": num_requests,
            "successful_requests": len(response_times),
            "failed_requests": errors,
            "avg_response_time_ms": statistics.mean(response_times),
            "median_response_time_ms": statistics.median(response_times),
            "min_response_time_ms": min(response_times),
            "max_response_time_ms": max(response_times),
            "success_rate": len(response_times) / num_requests * 100
        }
    else:
        return {"error": "All requests failed"}

# Usage
benchmark_results = benchmark_api_performance(
    "http://localhost:8000", 
    "demo-api-key-for-testing",
    num_requests=50
)
print(f"Average response time: {benchmark_results.get('avg_response_time_ms', 'N/A'):.2f}ms")
```

## Common Property Types and Sectors

Use these common values for testing:

### Property Types
- `"casa"` - House
- `"departamento"` - Apartment

### Common Sectors (Santiago, Chile)
- `"vitacura"`
- `"las condes"`
- `"providencia"`
- `"ñuñoa"`
- `"la reina"`
- `"lo barnechea"`
- `"santiago"`
- `"san miguel"`

### Typical Value Ranges
- **net_usable_area**: 30-500 m²
- **net_area**: 40-1000 m²
- **n_rooms**: 1-8 rooms
- **n_bathroom**: 1-6 bathrooms
- **latitude**: -33.7 to -33.2 (Santiago area)
- **longitude**: -70.8 to -70.4 (Santiago area)
