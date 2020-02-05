FROM centos:7
MAINTAINER Dilawar Singh <dilawars@ncbs.res.in>

# Install dependencies.
RUN yum -y update && yum -y install epel-release && yum -y update \
      && yum -y clean all --enablerepo='*'
RUN yum install -y git cmake3 gcc gcc-c++ make \
      python3 python3-devel python3-setuptools python3-numpy \
      && yum -y clean all --enablerepo='*'
RUN yum install -y gsl-devel  \
      && yum -y clean all --enablerepo='*'
# These are required to run tests.
RUN yum install -y python3-matplotlib python3-networkx graphviz python3-scipy  \
      && yum -y clean all --enablerepo='*'
RUN ln -s /usr/bin/cmake3 /usr/bin/cmake 
WORKDIR /home/root
# RUN git clone  https://github.com/dilawar/moose-core -b devel
# Run docker build from outside 
COPY . moose-core
RUN cd moose-core && python3 setup.py build test install 
CMD ["/usr/bin/python3", "-c", "'import moose;moose.test()'"]
