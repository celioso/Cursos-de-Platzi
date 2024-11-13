
FROM python:3.7
COPY . /app
RUN apt-get update && yes | apt-get upgrade
WORKDIR /app
RUN mkdir -p /tensorflow/models
RUN apt-get install -y git python3-pip
RUN pip install --upgrade pip
RUN pip install tensorflow
RUN apt-get install -y protobuf-compiler python3-pil python3-lxml
RUN pip install matplotlib
RUN git clone https://github.com/tensorflow/models.git /tensorflow/models
RUN apt-get update && apt-get install -y cmake
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /tensorflow/models/research
RUN protoc object_detection/protos/*.proto --python_out=.
RUN export PYTHONPATH=$PYTHONPATH:pwd:pwd/slim

WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python3", "./app.py"]
