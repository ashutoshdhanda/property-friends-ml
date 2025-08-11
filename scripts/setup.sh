#!/bin/bash
# Setup script for Property Friends ML project

set -e

echo "🏠 Setting up Property Friends ML Project..."

# Check if Python 3.8+ is available
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ is required. Found: $python_version"
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo "🔧 Installing package in development mode..."
pip install -e .

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p models data logs

# Copy environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "🔐 Creating environment file..."
    cp env.example .env
    echo "⚠️  Please edit .env file with your configuration"
fi

# Make scripts executable
echo "🔨 Making scripts executable..."
chmod +x scripts/*.sh scripts/*.py

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Add your training data to data/ directory"
echo "3. Train the model: python scripts/train_model.py --train-data data/train.csv --test-data data/test.csv"
echo "4. Run the API: python scripts/run_api.py"
echo "5. Or use Docker: docker-compose up"
echo ""
echo "📖 See README.md for detailed instructions"
