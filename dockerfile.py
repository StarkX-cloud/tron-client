FROM python:3.12

WORKDIR /app

COPY . .

RUN pip install fastapi uvicorn requests streamlit pandas pydantic

EXPOSE 9000

CMD ["python", "queue_server.py"]