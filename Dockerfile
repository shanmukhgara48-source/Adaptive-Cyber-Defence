FROM python:3.11-slim
WORKDIR /
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /adaptive_cyber_defense/
WORKDIR /adaptive_cyber_defense
EXPOSE 7860
CMD ["streamlit", "run", "ui.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
