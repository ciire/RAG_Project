# 1. Start with a Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies for 'unstructured' 
# (This handles the PDF/Markdown parsing libraries that Linux needs)
RUN apt-get update && apt-get install -y \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your project files
COPY . .

# 6. Set environment variables (prevents Python from buffering logs)
ENV PYTHONUNBUFFERED=1

# 7. Command to run your app
CMD ["python", "src/main.py"]