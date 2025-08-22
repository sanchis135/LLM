FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV OLLAMA_MODEL=llama3.1
ENV PROMPT_STYLE=strict
ENV PYTHONUNBUFFERED=1

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ENABLEXSRSFPROTECTION=false

EXPOSE 8501

RUN mkdir -p /app/data/kb /app/data/chroma /app/data/monitoring /app/reports

CMD ["streamlit", "run", "app/ui_streamlit/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
