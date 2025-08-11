# Property Friends ML API 🏠

A production-ready machine learning API for Chilean real estate property valuation. This project transforms a Jupyter notebook-based model into a scalable, containerized web service with comprehensive logging, authentication, and monitoring capabilities.

## 🎯 Overview

This API predicts property prices in Chile based on features like property type, location, size, and amenities. Built with FastAPI, Docker, and scikit-learn, it provides a robust foundation for real estate valuation services.

### Key Features

- 🚀 **Production-Ready**: Fully containerized with Docker and comprehensive logging
- 🔐 **Secure**: API key authentication with JWT token support
- 📊 **Monitoring**: Structured logging and health checks for operational monitoring
- 🔧 **Modular**: Clean architecture with data abstraction for future database integration
- 📖 **Well-Documented**: OpenAPI/Swagger documentation with interactive endpoints
- 🎯 **High Performance**: Efficient gradient boosting model with preprocessing pipeline

## 📁 Project Structure

```
property-friends-ml/
├── src/                     # Source code
│   ├── api/                 # FastAPI application
│   │   ├── main.py         # Main application
│   │   ├── endpoints.py    # API endpoints
│   │   └── auth.py         # Authentication
│   ├── models/             # ML model components
│   │   ├── model.py        # Main model class
│   │   ├── preprocessor.py # Data preprocessing
│   │   └── train.py        # Training script
│   ├── data/               # Data handling abstraction
│   │   └── loader.py       # Data loading strategies
│   └── utils/              # Utilities
│       ├── config.py       # Configuration management
│       ├── logger.py       # Structured logging
│       └── validation.py   # Input validation
├── scripts/                # Utility scripts
├── tests/                  # Test suite
├── docker/                 # Docker configuration
├── docs/                   # Documentation
├── models/                 # Trained models (gitignored)
├── data/                   # Data files (gitignored)
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)

1. **Clone and setup**:
   ```bash
   git clone https://github.com/ashutoshdhanda/property-friends-ml.git
   cd property-friends-ml
   cp env.example .env
   # Edit .env with your configuration
   ```

2. **Add your data**:
   ```bash
   # Place your CSV files in the project root
   cp /path/to/your/train.csv .
   cp /path/to/your/test.csv .
   ```

3. **Clean the data (if needed)**:
   ```bash
   # Run data cleaning to fix common issues
   python scripts/clean_data.py
   ```

4. **Train the model**:
   ```bash
   # Use cleaned data if you ran the cleaning script
   docker-compose run property-ml-api python scripts/train_model.py \
     --train-data train_clean.csv \
     --test-data test_clean.csv
   
   # Or use original data if it's clean
   docker-compose run property-ml-api python scripts/train_model.py \
     --train-data train.csv \
     --test-data test.csv
   ```

5. **Start the API**:
   ```bash
   docker-compose up
   ```

6. **Test the API**:
   ```bash
   curl http://localhost:8000/docs
   ```

### Option 2: Local Development

1. **Setup environment**:
   ```bash
   git clone https://github.com/ashutoshdhanda/property-friends-ml.git
   cd property-friends-ml
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

2. **Activate virtual environment**:
   ```bash
   source venv/bin/activate
   ```

3. **Train the model**:
   ```bash
   # Clean data if needed (run this if you encounter validation errors)
   python scripts/clean_data.py
   
   # Train with cleaned data (recommended)
   python scripts/train_model.py --train-data train_clean.csv --test-data test_clean.csv
   
   # Or train with original data if it's clean
   python scripts/train_model.py --train-data train.csv --test-data test.csv
   ```

4. **Start the API**:
   ```bash
   python scripts/run_api.py
   ```

## 🔑 Authentication

The API uses Bearer token authentication with API keys:

```bash
# Default demo API key for testing
API_KEY="demo-api-key-for-testing"

# Make authenticated requests
curl -H "Authorization: Bearer demo-api-key-for-testing" \
     http://localhost:8000/api/v1/health
```

## 📊 API Endpoints

### Core Endpoints

- **POST `/api/v1/predict`** - Predict single property price
- **POST `/api/v1/predict/batch`** - Batch prediction for multiple properties
- **GET `/api/v1/model/info`** - Model information and metrics
- **GET `/api/v1/health`** - Health check

### Documentation

- **GET `/docs`** - Interactive Swagger UI
- **GET `/redoc`** - ReDoc documentation
- **GET `/`** - API information and demo credentials

## 🏠 Property Features

The model expects these property features:

| Feature | Type | Description | Example |
|---------|------|-------------|---------|
| `type` | string | Property type | "casa" or "departamento" |
| `sector` | string | Neighborhood/sector | "vitacura", "las condes" |
| `net_usable_area` | float | Usable area (m²) | 140.0 |
| `net_area` | float | Total area (m²) | 170.0 |
| `n_rooms` | integer | Number of rooms | 4 |
| `n_bathroom` | integer | Number of bathrooms | 3 |
| `latitude` | float | Property latitude | -33.40123 |
| `longitude` | float | Property longitude | -70.58056 |

## 💡 Usage Examples

### Single Prediction

```python
import requests

url = "http://localhost:8000/api/v1/predict"
headers = {"Authorization": "Bearer demo-api-key-for-testing"}
data = {
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
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
# Output: {"prediction": 15847.23, "model_version": "1.0.0", "request_id": "..."}
```

### Batch Prediction

```python
import requests

url = "http://localhost:8000/api/v1/predict/batch"
headers = {"Authorization": "Bearer demo-api-key-for-testing"}
data = [
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
]

response = requests.post(url, json=data, headers=headers)
predictions = response.json()
for pred in predictions:
    print(f"Prediction: ${pred['prediction']:.2f}")
```

## ⚙️ Configuration

Configure the application via environment variables (`.env` file):

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# Security
API_KEY_SECRET=your-secret-key-here
SECRET_KEY=your-jwt-secret-key-here

# Model Configuration
MODEL_PATH=models/property_model.pkl
MODEL_VERSION=1.0.0

# Data Configuration
DATA_SOURCE=csv  # csv or database (future)
TRAIN_DATA_PATH=data/train.csv
TEST_DATA_PATH=data/test.csv

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Database (Future Use)
DATABASE_URL=postgresql://user:password@localhost:5432/property_db
```

## 🔄 Model Training

### Training Script

```bash
# Basic training
python scripts/train_model.py \
  --train-data data/train.csv \
  --test-data data/test.csv

# Custom output path
python scripts/train_model.py \
  --train-data data/train.csv \
  --test-data data/test.csv \
  --output models/my_model.pkl
```

### Training with Docker

```bash
# Train model in container
docker-compose run property-ml-api python scripts/train_model.py \
  --train-data data/train.csv \
  --test-data data/test.csv
```

## 📈 Model Performance

Current model metrics on test data:
- **RMSE**: 10,254 CLP
- **MAPE**: 40.04%
- **MAE**: 5,859 CLP

The model uses Gradient Boosting Regression with:
- Target encoding for categorical features
- 300 estimators with learning rate 0.01
- Max depth of 5
- Absolute error loss function

## 🔍 Monitoring & Logging

### Structured Logging

All API calls and predictions are logged with:
- Request/response details
- Processing times
- Feature values and predictions
- Error tracking
- Model performance metrics

### Health Checks

```bash
# API health
curl http://localhost:8000/api/v1/health

# Docker health check
docker-compose ps
```

### Metrics (Optional)

Enable Prometheus monitoring:

```bash
docker-compose --profile monitoring up
```

Access metrics at `http://localhost:9090`

## 🗄️ Future Database Integration

The project includes abstractions for future database integration:

```python
# Switch from CSV to database
# In .env file:
DATA_SOURCE=database
DATABASE_URL=postgresql://user:pass@localhost:5432/property_db

# The DataLoaderFactory will automatically use DatabaseDataLoader
# No code changes required in the API
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=src

# Run specific test file
pytest tests/test_api.py -v
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Pydantic Import Error
**Error:** `PydanticImportError: BaseSettings has been moved to the pydantic-settings package`

**Solution:** This is already fixed in the codebase. If you encounter it:
```bash
pip install pydantic-settings>=2.0.0
```

#### 2. Scikit-learn Compatibility Error
**Error:** `AttributeError: 'super' object has no attribute '__sklearn_tags__'`

**Solution:** This is already fixed by using compatible versions. The project uses:
```bash
scikit-learn==1.3.2  # Compatible version
# category-encoders removed (replaced with sklearn.preprocessing.OrdinalEncoder)
```

#### 3. Data Validation Errors
**Error:** `Training data validation failed: ['Non-positive values found in ...]`

**Solution:** Clean your data first:
```bash
# Run the data cleaning script
python scripts/clean_data.py

# Then train with cleaned data
python scripts/train_model.py --train-data train_clean.csv --test-data test_clean.csv
```

#### 4. Model File Not Found
**Error:** `Model not available: [Errno 2] No such file or directory: 'models/property_model.pkl'`

**Solution:** Train the model first:
```bash
# Ensure you have train.csv and test.csv in the project root
python scripts/train_model.py --train-data train.csv --test-data test.csv

# Or use cleaned data if validation fails
python scripts/clean_data.py
python scripts/train_model.py --train-data train_clean.csv --test-data test_clean.csv
```

#### 5. Port Already in Use
**Error:** `[Errno 48] Address already in use`

**Solution:** Change the port or kill existing process:
```bash
# Check what's using port 8000
lsof -i :8000

# Kill the process (replace PID with actual process ID)
kill -9 <PID>

# Or use a different port in .env
API_PORT=8001
```

#### 6. Docker Build Issues
**Error:** Docker build failures

**Solution:** 
```bash
# Clean Docker cache and rebuild
docker system prune -a
docker-compose build --no-cache
docker-compose up
```

### Data Requirements

Your CSV files should have these columns:
- `type`: "casa" or "departamento"
- `sector`: neighborhood name (e.g., "vitacura", "las condes")
- `net_usable_area`: positive number (square meters)
- `net_area`: positive number ≥ net_usable_area
- `n_rooms`: positive integer ≥ 1
- `n_bathroom`: positive integer ≥ 1
- `latitude`: number between -56 and -17 (Chile bounds)
- `longitude`: number between -76 and -66 (Chile bounds)
- `price`: positive number (Chilean Pesos)

**Note:** If your data doesn't meet these requirements, use the cleaning script:
```bash
python scripts/clean_data.py
```

## 🐳 Docker Commands

```bash
# Build and start services
docker-compose up --build

# Start with database
docker-compose --profile with-db up

# Start with monitoring
docker-compose --profile monitoring up

# View logs
docker-compose logs -f property-ml-api

# Scale API instances
docker-compose up --scale property-ml-api=3
```

## 📋 Model Assumptions & Limitations

### Data Assumptions
- Property prices are in Chilean Pesos (CLP)
- Coordinates are within Chile's geographic bounds
- Property types are limited to "casa" and "departamento"
- Areas are in square meters

### Model Limitations
- Trained on historical data - may not reflect current market conditions
- Limited to features available in training data
- Performance may degrade with properties significantly different from training set
- No handling for luxury/premium properties outside normal price ranges

### Areas for Improvement
1. **Feature Engineering**: Add more location-based features (distance to amenities, transportation)
2. **Model Ensemble**: Combine multiple algorithms for better performance
3. **Time Series**: Incorporate temporal trends and seasonality
4. **External Data**: Integrate economic indicators, market trends
5. **Advanced Validation**: Implement more sophisticated cross-validation strategies
6. **Outlier Detection**: Better handling of unusual properties
7. **Confidence Intervals**: Provide prediction uncertainty estimates

## 🔧 Development

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Features

1. **New Endpoints**: Add to `src/api/endpoints.py`
2. **Model Updates**: Modify `src/models/model.py`
3. **Data Sources**: Extend `src/data/loader.py`
4. **Configuration**: Update `src/utils/config.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 🙋‍♂️ Support

For questions and support:

1. Check the [API documentation](http://localhost:8000/docs) when running
2. Review the logs for error details
3. Open an issue on GitHub
4. Check the health endpoint: `/api/v1/health`

## 🚧 Roadmap

- [ ] Database integration (PostgreSQL)
- [ ] Advanced monitoring dashboard
- [ ] A/B testing framework for model versions
- [ ] Automated model retraining pipeline
- [ ] Integration with external property data sources
- [ ] Mobile app API endpoints
- [ ] Real-time price update notifications

---

Ashutosh Dhanda