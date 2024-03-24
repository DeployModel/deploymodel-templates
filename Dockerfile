FROM python:3.9

WORKDIR /app/
COPY ./requirements.txt .
RUN pip install -r requirements.txt

ENV PYTHONPATH=/app
COPY . .
WORKDIR /app
ENTRYPOINT ["python", "deploy.py"]