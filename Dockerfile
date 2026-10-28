# The API, console and live-lab controller. Copies explicit paths only, so
# local secrets (.env), raw data and working notes never enter the image.
FROM python:3.13-slim

WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY src/ src/
COPY static/ static/
COPY runs/replay/ runs/replay/
COPY runs/detection.jsonl runs/explain_summary.jsonl runs/trust_summary.jsonl runs/

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
