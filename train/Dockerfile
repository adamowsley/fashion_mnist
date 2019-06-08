FROM tensorflow/tensorflow:2.0.0b0-gpu-py3

LABEL MAINTAINER="Adam Owsley <adam.owsley@gmail.com>"

COPY requirements.txt /opt/app/requirements.txt

WORKDIR /opt/app

RUN pip3 install -r requirements.txt

COPY . /opt/app

ENV PYTHON_PACKAGES="numpy matplotlib"

CMD ["python3", "fashion_train.py"]
