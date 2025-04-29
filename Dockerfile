FROM python:3.10
WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["python", "HousePricePrediction.py"]