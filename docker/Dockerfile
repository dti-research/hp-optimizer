FROM ubuntu:18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         python3-dev \ 
         python3-pip \
         python3-setuptools \
         python-opengl \
         libopenmpi-dev \
         openmpi-bin \
         openssh-client \
         openssh-server \
         net-tools \
         iputils-ping \
         ca-certificates \
         libglib2.0-0 \
         libsm6 libxext6 libxrender-dev \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip3 install scipy \
         opencv-python \
         numpy \
         gym

RUN pip3 install hpbandster \
         tensorflow \
         hyperopt

RUN pip3 install stable_baselines

RUN git clone https://github.com/dti-research/hp-optimizer.git

