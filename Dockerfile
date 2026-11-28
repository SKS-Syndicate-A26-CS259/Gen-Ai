FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/huggingface

WORKDIR /code

COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /code/requirements.txt

COPY app/ /code/app/
COPY vectorstore/ /code/vectorstore/
COPY Model/ /code/Model/

# Izin tulis untuk cache HuggingFace & SQLite WAL ChromaDB (kompatibel dengan non-root container seperti HF Spaces)
RUN mkdir -p /tmp/huggingface && chmod -R 777 /code/vectorstore /tmp/huggingface

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
