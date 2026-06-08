FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . ./

ENV PYTHONUNBUFFERED=1
EXPOSE 9000

CMD ["python", "queue_server.py"]
