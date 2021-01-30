FROM nvcr.io/nvidia/tensorflow:20.09-tf1-py3

# Install dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y \
    libgl1-mesa-glx \
    python3-pyqt5 \
    python3-tk

# Install Python dependencies
WORKDIR /tmp
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install pycuda

# Add project code to PYTHONPATH (assumes volume mount point is /code)
ENV PYTHONPATH="/code:$PYTHONPATH"

WORKDIR /code
