FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY src/ ./src/
COPY tests/ ./tests/

# No secrets are baked in; pass LLM_API_KEY at run time with -e.
EXPOSE 5000

# Threads, not processes: each request spends most of its time waiting on the
# LLM provider, and the LP solve is milliseconds.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "8", \
     "--timeout", "60", "--access-logfile", "-", "main:app"]
