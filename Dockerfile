FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the package
COPY llm_training/ ./llm_training/
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .

# Install the package
RUN pip install --no-cache-dir -e .

# Set Python path
ENV PYTHONPATH=/app

CMD ["/bin/bash"]
