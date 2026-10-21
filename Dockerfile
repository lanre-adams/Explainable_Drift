# BASE and SYSTEM_DEPS exist only so the image can be tested offline; normal builds use the defaults.
ARG BASE=python:3.12-slim
FROM ${BASE}
ARG SYSTEM_DEPS=1
ARG PIP_EXTRA=""

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl
RUN if [ "$SYSTEM_DEPS" = "1" ]; then apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*; fi

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir $PIP_EXTRA -r requirements.txt

COPY data/ data/
COPY app/ app/
RUN useradd -m pilot && mkdir -p /app/runs && chown -R pilot /app/runs
USER pilot

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -fs http://localhost:8000/api/health || exit 1
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
