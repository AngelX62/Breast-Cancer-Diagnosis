# Start from Python's Base Image
FROM python:3.12

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# scikit-learn runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
    
    
# Change working directory
WORKDIR /app

# Add requirements file to image
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what server needs
COPY app/ app/
COPY models/ models/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]