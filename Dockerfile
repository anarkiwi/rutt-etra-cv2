FROM python:3.12-slim AS deps
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt ./
RUN pip install --no-cache-dir opencv-python-headless && \
    grep -v '^opencv-python$' requirements.txt > runtime.txt && \
    pip install --no-cache-dir -r runtime.txt

FROM deps AS devdeps
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

FROM python:3.12-slim AS test
COPY --from=devdeps /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY pyproject.toml rutt-etra.py rutt-scope.py rutt-laser.py ./
COPY ruttetra ./ruttetra
COPY tests ./tests
COPY tools ./tools
CMD ["pytest"]

FROM python:3.12-slim AS runtime
COPY --from=deps /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY pyproject.toml rutt-etra.py rutt-scope.py rutt-laser.py ./
COPY ruttetra ./ruttetra
ENTRYPOINT ["python"]
CMD ["rutt-etra.py", "--help"]
