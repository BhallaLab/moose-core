FROM quay.io/pypa/manylinux2014_x86_64

ARG PYPI_PASSWORD

MAINTAINER Dilawar Singh <dilawar.s.rajput@gmail.com>
ENV PATH=/usr/local/bin:$PATH
RUN yum update -y
RUN yum install -y cmake3
RUN yum install -y wget  
RUN ln -s /usr/bin/cmake3 /usr/bin/cmake
RUN ln -s /usr/bin/ctest3 /usr/bin/ctest
COPY ../wheel /root/
WORKDIR /root
RUN cd wheel && ./build_wheels_linux.sh 
RUN echo "pass $PYPI_PASSWORD"
RUN cd wheel && ./test_and_upload.sh "$PYPI_PASSWORD"
