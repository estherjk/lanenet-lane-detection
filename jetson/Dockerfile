# Docker container with CUDA-enabled OpenCV for NVIDIA Jetson
# Source: https://github.com/mdegans/nano_build_opencv/blob/docker/Dockerfile

ARG JETPACK_VERSION="r32.4.3"

FROM registry.hub.docker.com/mdegans/l4t-base:${JETPACK_VERSION}

# change these here or with --build-arg FOO="BAR" at build time
ARG OPENCV_VERSION="4.4.0"
ARG OPENCV_DO_TEST="FALSE"
# note: 8 jobs will fail on Nano. Try 1 instead.
ARG OPENCV_BUILD_JOBS="8"
# required for apt-get -y to work properly:
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /usr/local/src/build_opencv

COPY build_opencv.sh .

RUN /bin/bash build_opencv.sh

# Install dependencies for LaneNet
RUN set -xe \
    && apt-get update \
    && apt-get install -y python3-pip
RUN pip3 install --upgrade pip

WORKDIR /tmp
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Add project code to PYTHONPATH (assumes volume mount point is /code)
ENV PYTHONPATH="/code:$PYTHONPATH"

WORKDIR /code
