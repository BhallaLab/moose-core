NAME:=dilawars/moose-travis
VERSION:=$(shell date +"%Y%m%d")

PROJECT_ROOT_DIR:=(PWD)/../..

all : build

build : Dockerfile
	docker build -t $(NAME):$(VERSION) .
	docker build -t $(NAME):latest .

upload :
	docker push $(NAME):$(VERSION)
	docker push $(NAME):latest

run :
	docker run -it $(NAME):latest bash

