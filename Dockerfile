FROM python:3.9.20-slim

RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --deploy --system

COPY ["startup_success_model.pkl", "status_label_encoder.pkl", "./"]
COPY [ "predict.py", "./" ]

EXPOSE 8080

ENTRYPOINT [ "gunicorn", "--bind", "0.0.0.0:8080", "predict:app" ]