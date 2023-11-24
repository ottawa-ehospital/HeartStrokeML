FROM python:3.10-slim-bullseye

WORKDIR /stroke

COPY ./requirements.txt /stroke/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . /stroke

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
