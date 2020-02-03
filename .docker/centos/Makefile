NAME:=dilawars/centos-moose
VERSION:=$(shell date +%Y.%m.%d)

all : build

build : Dockerfile
	cd ../../ && docker build -t $(NAME):$(VERSION) .docker/centos
	cd ../../ && docker build -t $(NAME):latest .docker/centos

upload :
	docker push $(NAME):$(VERSION)
	docker push $(NAME):latest

