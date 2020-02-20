FROM ubuntu:xenial
MAINTAINER Dilawar Singh <dilawars@ncbs.res.in>

# Install dependencies.
RUN apt update && apt install -y git cmake gcc g++ make vim-athena \
        libboost-all-dev libgsl-dev \
        && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y python3-pip python-pip \
        python3-numpy python-numpy \
        python3-matplotlib python-matplotlib \
        && rm -rf /var/lib/apt/lists/*

RUN git config --global user.name "Dilawar Singh" \
  && git config --global user.email "dilawars@ncbs.res.in"
RUN git clone https://github.com/dilawar/vim ~/.vim -b minimal --depth 1
WORKDIR /root
COPY . moose-core/
RUN ls -lta
CMD cd moose-core && ./.ci/travis_build_linux.sh
