# Differential Diagnosis Backend

A FastAPI-based backend for the differential diagnosis application.

## Setup

### With uv (recommended)

```bash
# Install dependencies
uv sync

# Run the development server
uv run uvicorn app.main:app --reload
```

### With pip (legacy)

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the development server
uvicorn app.main:app --reload
```

## API Endpoints

- `GET /` - Hello World
- `GET /api/users/{user_id}` - Get user by ID
