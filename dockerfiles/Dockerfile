FROM tensorflow/tensorflow:latest-gpu
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && apt-get install -y sudo wget git nano python3-matplotlib graphviz \
    # OpenCV libraries
    build-essential cmake pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev \
    libtbb2 libtbb-dev libdc1394-22-dev \
    && pip3 install pydot \
    # Install OpenCV
    && mkdir /home/opencv_build && cd /home/opencv_build \
    && git clone https://github.com/opencv/opencv.git \
    && git clone https://github.com/opencv/opencv_contrib.git \
    && cd opencv && mkdir build && cd build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=/home/opencv_build/opencv_contrib/modules \
    -D BUILD_EXAMPLES=ON .. \
    && make -j8 && make install