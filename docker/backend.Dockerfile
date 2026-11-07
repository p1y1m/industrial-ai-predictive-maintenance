FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN groupadd --system app && useradd --system --gid app app
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p src/industrial_pm && touch src/industrial_pm/__init__.py \
    && pip install 'torch>=2.4,<3' --index-url https://download.pytorch.org/whl/cpu \
    && pip install '.[mlops]' \
    && rm -rf src
COPY src ./src
ENV PYTHONPATH=/app/src
COPY configs ./configs
USER app
EXPOSE 8000
CMD ["uvicorn","industrial_pm.api.main:app","--host","0.0.0.0","--port","8000"]
