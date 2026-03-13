FROM python:3.14

WORKDIR /sentiment-analysis

COPY ./requirements.txt /sentiment-analysis/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /sentiment-analysis/requirements.txt

COPY ./app /sentiment-analysis/app

# uvicorn app.main:app --host
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
