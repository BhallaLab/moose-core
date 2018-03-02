all : wheel
	echo "Done building wheel"
	echo "Upload on pypy? "

wheel : ./Dockerfile ./build_wheels.sh
	echo "If following step is not doing anything \
	    check if http_proxy is set/unset in Dockerfile"
	docker build -t bhallalab/pymoose-wheel:3.2.0 .

