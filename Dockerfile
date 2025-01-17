FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04
MAINTAINER lertlove@gmail.com

# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV TZ=Asia/Bangkok \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget

RUN apt install -y python python3 python3-pip
RUN unlink /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python

ENV DEBIAN_FRONTEND=noninteractive
RUN apt install -y python3-opencv

RUN python -m pip install --upgrade pip
RUN apt install -y pkg-config libhdf5-dev

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
#  --extra-index-url https://download.pytorch.org/whl/cu113

RUN apt install -y python3-tk
# RUN apt install -y openjdk-8-jdk-headless -qq 
RUN yes|pip install pyzmq


CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root","--NotebookApp.token=''","--NotebookApp.password=''"]
