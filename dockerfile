FROM python:3.9-slim

WORKDIR /usr/src/app

COPY . .

RUN pip install poetry

RUN poetry config virtualenvs.create false \
    && poetry install --no-dev  # Avoid installing dev dependencies

EXPOSE 80

ENV NAME World

CMD ["python", "app.py"]
