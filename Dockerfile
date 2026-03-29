FROM python:3.11-slim
WORKDIR /
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /adaptive_cyber_defense/
WORKDIR /adaptive_cyber_defense
EXPOSE 8501
CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
