FROM nvcr.io/nvidia/l4t-tensorflow:r32.4.3-tf1.15-py3

# Install dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -y git \
    build-essential \
    cmake \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer1.0-dev \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    pkg-config

### BEGIN: Install OpenCV ###

RUN mkdir /opencv
WORKDIR /opencv

# Download OpenCV sources
RUN git clone https://github.com/opencv/opencv.git --single-branch --branch 4.4.0 && \
    git clone https://github.com/opencv/opencv_contrib.git --single-branch --branch 4.4.0

# Create build directory
RUN mkdir build
WORKDIR /opencv/build

# NOTE: Unable to build OpenCV w/ CUDA for now...
RUN cmake -D BUILD_EXAMPLES=OFF \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_opencv_python3=ON \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_TESTS=OFF \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/opencv/opencv_contrib/modules \
        -D WITH_CUDA=OFF \
        -D WITH_GSTREAMER=ON \
        -D WITH_LIBV4L=ON \
        -D WITH_OPENGL=ON \
        ../opencv

# Make! Use -j1 if running on a more compute-constrained device, e.g. Nano w/ SD card
RUN make -j4 && \
    make install

### END: Install OpenCV ###

### BEGIN: Install dependencies for LaneNet ###

RUN apt install -y \
    libatlas-base-dev \
    gfortran \
    python3-matplotlib

# Install Python dependencies
WORKDIR /tmp
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install pycuda

# Add project code to PYTHONPATH (assumes volume mount point is /code)
ENV PYTHONPATH="/code:$PYTHONPATH"

### END: Install dependencies for LaneNet ###

WORKDIR /code
