# 1. Use an official lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies for FAISS and PDF processing
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy only requirements first to leverage Docker caching
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your application code
COPY . .

# 7. Create the temporary directory for document uploads
RUN mkdir -p temp_docs && chmod 777 temp_docs

# 8. Expose the default Streamlit port
EXPOSE 8501

# 9. Healthcheck to ensure the app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 10. Start the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
