FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Create log directories
RUN mkdir -p outputs/logs outputs/evals

ENV HOST=0.0.0.0
ENV PORT=8000
ENV WORKERS=1
ENV MAX_STEPS=20
ENV EPISODE_TIMEOUT=300
ENV SUSPICION_THRESHOLD=0.7
ENV LOG_DIR=outputs/logs

EXPOSE 8000

CMD ["sh", "-c", "uvicorn server.app:app --host $HOST --port $PORT --workers $WORKERS"]
