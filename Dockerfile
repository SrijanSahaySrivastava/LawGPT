FROM python:3.11

WORKDIR /app

COPY requirements.txt /app/requirements.txt


RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
