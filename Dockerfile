FROM python:3.11-slim
WORKDIR /
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY . /adaptive_cyber_defense/
WORKDIR /adaptive_cyber_defense
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:7860/_stcore/health || exit 1
CMD ["streamlit", "run", "ui.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
