# PyAtonAPI

A FastAPI application for maritime AtoN (Aids to Navigation) data analytics with LLM-powered natural language querying.

## Features

- Real-time AtoN data from ClickHouse database
- Natural language to SQL query generation using Ollama (llama3)
- In-memory analytics with DuckDB
- REST API endpoints for AtoN monitoring

## Tech Stack

- **FastAPI** - Web framework
- **Ollama** - Local LLM (llama3)
- **ClickHouse** - Time-series database
- **DuckDB** - In-memory SQL analytics
- **Pandas/NumPy** - Data processing

## Prerequisites

- Docker & Docker Compose

## Quick Start

### 1. Clone and navigate to the project

```bash
cd PyAtonAPI
```

### 2. Start the services

```bash
docker-compose up -d
```

### 3. Pull the llama3 model (first time only)

```bash
docker-compose exec ollama ollama pull llama3
```

### 4. Verify services are running

```bash
docker-compose ps
```

### 5. Access the API

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /aton/all` | Get all AtoN data |
| `GET /aton/summary` | Get AtoN summary by MMSI or date |
| `GET /atonvolt_mmsi/{mmsi}` | Get voltage data by MMSI |
| `GET /atonheartbeat_mmsi/{mmsi}` | Get heartbeat data by MMSI |
| `GET /aton_by_month/{month}` | Get AtoN data by month |
| `GET /llm_interactive` | Natural language query for AtoN summary |
| `GET /llm_query_analytics` | Natural language query for real-time analytics |

## Docker Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up -d --build

# View API logs
docker-compose logs -f api

# View Ollama logs
docker-compose logs -f ollama

# Check loaded models
docker-compose exec ollama ollama list
```

## Local Development (without Docker)

### 1. Create virtual environment

```bash
python -m venv env
source env/bin/activate  # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Ollama locally

```bash
ollama serve
ollama pull llama3
```

### 4. Run the API

```bash
uvicorn endpoints:app --reload --host 0.0.0.0 --port 8000
```

## Project Structure

```
PyAtonAPI/
├── config.py           # Configuration constants
├── data_loader.py      # ClickHouse data loading functions
├── endpoints.py        # FastAPI routes
├── llm_search.py       # LLM query generation
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker image definition
├── docker-compose.yml  # Multi-container setup
└── README.md
```

## License

Private
