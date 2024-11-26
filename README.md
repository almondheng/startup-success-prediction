# Project

## Problem Description

Startups often face high uncertainty, and understanding the factors that lead to success or failure can be invaluable for entrepreneurs and investors. By analyzing the dataset from Crunchbase, we can identify key characteristics such as funding amounts, industry sectors, and geographical locations that distinguish successful startups from those that fail.

A machine learning model can be trained on this historical startup dataset to predict the likely outcome of a startup. The predictions can potentially help venture capitals and angel investors in making investment decisions.

Dataset source: https://www.kaggle.com/datasets/yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase

# Getting started

## Prerequisite
- Python 3.9.x
- Pipenv
- Docker

## Installation
```bash
pipenv install
pipenv shell
```

## Run Training
```bash
python train.py
```

## Start application
### Local
```bash
waitress-serve --host 127.0.0.1 predict:app
```

### Docker
```bash
docker build -t startup-success-predictor .
docker run -p 8080:8080 startup-success-predictor
```

## Run inference
```bash
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{
         "funding_total_usd": 25500000,
         "funding_rounds": 10,
         "country_code": "US",
         "state_code": "CA",
         "region": "Silicon Valley",
         "city": "San Francisco",
         "category_list": "Technology|Software",
         "first_funding_at": "2022-01-01",
         "last_funding_at": "2023-01-01"
     }'
```