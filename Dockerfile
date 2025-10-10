# syntax=docker/dockerfile:1
FROM python:3.11-slim

# (optional) system deps you might need later
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# install your package with dev+docs extras
RUN python -m pip install --upgrade pip && \
    pip install -e .[dev,docs]

# default: show CLI help (override in docker run)
CMD ["icdm2025-demo", "--help"]
