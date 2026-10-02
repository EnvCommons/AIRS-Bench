FROM python:3.11-slim

RUN apt update && apt upgrade -y && apt install -y curl

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY airs_bench.py .
COPY evaluate.py .
COPY task_config.py .
COPY apps_eval/ apps_eval/

EXPOSE 8000

CMD ["python", "server.py"]
