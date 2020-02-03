FROM centos:7
MAINTAINER Dilawar Singh <dilawars@ncbs.res.in>

# Install dependencies.
RUN yum -y update && yum -y install epel-release && yum -y update \
      && yum -y clean all --enablerepo='*'
RUN yum install -y git cmake gcc gcc-c++ make \
      python3 python3-setuptools gsl-dev python3-numpy \
      && yum -y clean all --enablerepo='*'

WORKDIR /home/root
# RUN git clone  https://github.com/dilawar/moose-core -b devel
# Run docker build from outside 
COPY . /home/root/moose-core
RUN cd moose-core && python3 setup.py build
