.PHONY: prepare run

prepare:
	@if [ ! -f ./include/json.hpp ]; then \
		mkdir -p ./include; \
		wget -O ./include/json.hpp \
			https://github.com/nlohmann/json/releases/latest/download/json.hpp; \
	fi
	@mkdir -p ./build

run: prepare
	g++ -std=c++20 -g -O2 \
		-I./include \
		./main.cpp \
		./src/*.cpp \
		-o ./build/main \
		&& ./build/main

