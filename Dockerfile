FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
MAINTAINER lertluck.l@obodroid.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget

RUN apt install -y python python3 python-pip python3-pip
RUN unlink /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root","--NotebookApp.token=''","--NotebookApp.password=''"]
