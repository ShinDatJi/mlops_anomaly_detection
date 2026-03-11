FROM apache/airflow:2.11.1

USER root

RUN apt-get update \
    && apt-get install -y docker-compose-plugin \
    && apt-get clean

USER airflow

# RUN pip install apache-airflow-providers-docker==4.5.2
