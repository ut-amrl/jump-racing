# Clang is a good compiler to use during development due to its faster compile
# times and more readable output.
# C_compiler=/usr/bin/clang
# CXX_compiler=/usr/bin/clang++

# GCC is better for release mode due to the speed of its output, and its support
# for OpenMP.
C_compiler=/usr/bin/gcc
CXX_compiler=/usr/bin/g++

#acceptable build_types: Release/Debug/Profile
build_type=Release
# build_type=Debug

.SILENT:
all: build/CMakeLists.txt.copy
	$(info Build_type is [${build_type}])
	$(MAKE) --no-print-directory -C build

docker_all: docker_build_q
	docker run --rm --volume "$(shell pwd)":/home/dev/jump_racing jump_racing "cd jump_racing && make -j"

docker_shell: docker_build_q
	if [ $(shell docker ps -a -f name=jump_racing_shell | wc -l) -ne 2 ]; then docker run -dit --name jump_racing_shell --volume "$(shell pwd)":/home/dev/jump_racing --workdir /home/dev/jump_racing -p 10272:10272 jump_racing; fi
	docker exec -it jump_racing_shell bash -l

docker_stop:
	docker container stop jump_racing_shell
	docker container rm jump_racing_shell

docker_build:
	docker build --build-arg HOST_UID=$(shell id -u) -t jump_racing .

docker_build_q:
	docker build -q --build-arg HOST_UID=$(shell id -u) -t jump_racing .

# Sets the build type to Debug.
set_debug:
	$(eval build_type=Debug)

# Ensures that the build type is debug before running all target.
debug_all: | set_debug all

clean:
	rm -rf build bin lib

build/CMakeLists.txt.copy: CMakeLists.txt Makefile
	mkdir -p build
	cd build && cmake -DCMAKE_BUILD_TYPE=$(build_type) \
		-DCMAKE_CXX_COMPILER=$(CXX_compiler) \
		-DCMAKE_C_COMPILER=$(C_compiler) ..
	cp CMakeLists.txt build/CMakeLists.txt.copy
