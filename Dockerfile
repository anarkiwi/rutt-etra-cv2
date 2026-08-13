FROM python:3.12-slim AS deps
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir opencv-python-headless && \
    grep -v '^opencv-python$' requirements.txt > requirements-headless.txt && \
    pip install --no-cache-dir -r requirements-headless.txt -r requirements-dev.txt

FROM python:3.12-slim
COPY --from=deps /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY pyproject.toml rutt-etra.py ./
COPY ruttetra ./ruttetra
COPY tests ./tests
CMD ["pytest"]
