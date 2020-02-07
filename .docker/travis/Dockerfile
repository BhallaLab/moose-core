FROM ubuntu:xenial
MAINTAINER Dilawar Singh <dilawars@ncbs.res.in>

# Install dependencies.
RUN apt update && apt install -y git cmake gcc g++ make vim-athena \
      && rm -rf /var/lib/apt/lists/*

RUN git config --global user.name "Dilawar Singh" \
  && git config --global user.email "dilawars@ncbs.res.in"
RUN git clone https://github.com/dilawar/vim ~/.vim -b minimal --depth 1
WORKDIR /root
