@echo off
echo 🔬 Hybrid Search PoC - Quick Start
echo ==================================

REM Check if .env file exists
if not exist .env (
    echo ❌ .env file not found!
    echo Please create a .env file with your credentials:
    echo.
    echo OPENAI_API_KEY=your_openai_api_key_here
    echo NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
    echo NEO4J_USERNAME=neo4j
    echo NEO4J_PASSWORD=your_neo4j_password_here
    echo PINECONE_API_KEY=your_pinecone_api_key_here
    echo.
    pause
    exit /b 1
)

echo ✅ .env file found

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker not found! Please install Docker first.
    echo Falling back to local Python setup...
    
    REM Install Python dependencies
    echo 📦 Installing Python dependencies...
    pip install -r requirements.txt
    
    REM Run Streamlit
    echo 🚀 Starting Streamlit app...
    streamlit run app/demo.py
) else (
    echo 🐳 Docker found! Using Docker setup...
    
    REM Check if docker-compose is available
    docker-compose --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo 🚀 Starting with Docker Compose...
        docker-compose up --build
    ) else (
        echo 🚀 Starting with Docker...
        docker build -t hybrid-search-poc .
        docker run -p 8501:8501 --env-file .env hybrid-search-poc
    )
)

pause