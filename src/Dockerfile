FROM python:3.10-slim


WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV API_KEY=""

CMD ["python","main.py"]

