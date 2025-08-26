#!/bin/bash

# Quick start script for Hybrid Search PoC

echo "🔬 Hybrid Search PoC - Quick Start"
echo "=================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "Please create a .env file with your credentials:"
    echo ""
    echo "OPENAI_API_KEY=your_openai_api_key_here"
    echo "NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io"
    echo "NEO4J_USERNAME=neo4j"
    echo "NEO4J_PASSWORD=your_neo4j_password_here"
    echo "PINECONE_API_KEY=your_pinecone_api_key_here"
    echo ""
    exit 1
fi

echo "✅ .env file found"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found! Please install Docker first."
    echo "Falling back to local Python setup..."
    
    # Install Python dependencies
    echo "📦 Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Run Streamlit
    echo "🚀 Starting Streamlit app..."
    streamlit run app/demo.py
else
    echo "🐳 Docker found! Using Docker setup..."
    
    # Check if docker-compose is available
    if command -v docker-compose &> /dev/null; then
        echo "🚀 Starting with Docker Compose..."
        docker-compose up --build
    else
        echo "🚀 Starting with Docker..."
        docker build -t hybrid-search-poc .
        docker run -p 8501:8501 --env-file .env hybrid-search-poc
    fi
fi