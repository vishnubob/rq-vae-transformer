FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime
RUN apt-get update && \
    apt-get install -y git build-essential && \
    pip install -U pip
COPY / /rq-vae
WORKDIR /rq-vae
RUN pip install -r requirements.txt
