FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_HOME=/opt/poetry \
    POETRY_VERSION=2.2.1 \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_VIRTUALENVS_PATH=/opt/poetry-venvs

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${POETRY_HOME}/bin:${PATH}"
RUN chmod -R a+rX "${POETRY_HOME}" \
    && ln -sf "${POETRY_HOME}/bin/poetry" /usr/local/bin/poetry

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry install --all-groups --no-interaction --no-root
RUN chmod -R a+rX /opt/poetry-venvs

COPY src ./src
COPY tests ./tests
COPY dvc.yaml dvc.lock ./
COPY CONTRIBUTIONS.md README.md ./

# Keep default command lightweight; run pipeline commands explicitly.
CMD ["poetry", "run", "python", "-c", "print('EVBuddy pipeline image ready')"]
