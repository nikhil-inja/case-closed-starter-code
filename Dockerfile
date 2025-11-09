# Dockerfile for v29.py Flask agent

# Use Python 3.12 slim image for smaller size
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the agent file
COPY v29.py .

# Expose the port (default 5008, but can be overridden via PORT env var)
EXPOSE 5008

# Set environment variable for port (can be overridden)
ENV PORT=5008

# Run the Flask app
CMD ["python", "v29.py"]
