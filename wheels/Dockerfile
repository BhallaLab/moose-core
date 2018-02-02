FROM quay.io/pypa/manylinux1_x86_64
MAINTAINER Dilawar Singh <dilawar.s.rajput@gmail.com>

# If you are behind proxy,  uncomment the following lines with appropriate
# values. Otherwise comment them out.
ENV http_proxy http://proxy.ncbs.res.in:3128
ENV https_proxy http://proxy.ncbs.res.in:3128

RUN yum update
RUN yum install -y cmake28 && ln -s /usr/lib/cmake28 /usr/local/bin/cmake
RUN yum install -y wget 
RUN wget --no-check-certificate ftp://ftp.gnu.org/gnu/gsl/gsl-2.4.tar.gz 
RUN tar xvf gsl-2.4.tar.gz && cd gsl-2.4 && ./configure && make -j2 &&  make install && cd 
RUN yum install -y git
RUN git clone --depth 10 https://github.com/BhallaLab/moose-core 
RUN cd moose-core && git pull && cd wheels && ./build_wheels.sh
