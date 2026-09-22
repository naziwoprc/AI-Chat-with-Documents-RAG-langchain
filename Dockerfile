# ✅ Use Python 3.12 to match your local environment
FROM python:3.12-slim

# ✅ Set working directory
WORKDIR /app

# ✅ Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ✅ Copy requirements first (better caching)
COPY requirements.txt .

# ✅ Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ✅ Copy project files
COPY . .

# ✅ Expose FastAPI port
EXPOSE 8000

# ✅ Start FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]