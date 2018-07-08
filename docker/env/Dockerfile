FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        software-properties-common \
        curl \
        wget \
        git \
        unzip \
        cmake \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        libncursesw5-dev \
        libreadline-dev \
        libssl-dev \
        libgdbm-dev \
        libsqlite3-dev \
        libbz2-dev \
        tk-dev \
        zlib1g-dev \
        swig \
        libfreetype6-dev \
        libpng12-dev \
        libgdbm-dev \
        libjpeg8-dev \
        libtiff5-dev \
        libjasper-dev \
        libatlas-base-dev \
        libwebp-dev \
        libopenexr-dev \
        gfortran \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        vim \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
ENV PATH /root/.pyenv/shims:/root/.pyenv/bin:$PATH
ENV PYENV_SHELL bash
RUN command pyenv rehash 2>/dev/null
RUN git clone https://github.com/momo-lab/pyenv-install-latest.git /root/.pyenv/plugins/pyenv-install-latest

RUN pyenv install-latest 3.5
RUN pyenv global $(pyenv versions)
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

WORKDIR /root

ADD requirements.txt /tmp/requirements.txt
RUN pip --no-cache-dir install -r /tmp/requirements.txt

EXPOSE 6006

ARG UID
ENV NOTVISIBLE "in users profile"

RUN useradd -d /home/anon -ms /bin/bash -u $UID -G sudo anon
RUN echo anon:anon | chpasswd
RUN chmod -R 775 /root

USER anon
RUN mkdir /home/anon/crt-task

WORKDIR /home/anon/crt-task
