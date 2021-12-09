SRC = main.cpp
CXX = g++
CXXFLAGS = -O3 -std=c++14 -Ienv utils.cpp
LIBFLAGS = -static-libstdc++  -Wno-return-type-c-linkage -shared -fpic  -o
OPENMP = -fopenmp -Dparallel
#on M1 remove -static-libstdc++ and add --target=x86_64-apple-darwin
EXE = $(SRC:.cpp=.x)

.SUFFIXES:
SUFFIXES =

.SUFFIXES: .cpp .x

VPATH = env

all: $(EXE) libkite.so

parallel: $(OPENMP) libkite.so


.PHONY: all

%.x: %.cpp

	$(CXX) $< -o $@ $(CXXFLAGS)

libkite.so: libkite.cpp
			$(CXX) $(LIBFLAGS) $@ $< $(CXXFLAGS)


clean:
		rm -f $(EXE) *~
		rm libkite.so

.PHONY: clean

main.x: kite.hpp constants.hpp vect.hpp utils.cpp wind.hpp

libkite.so: kite.hpp constants.hpp vect.hpp utils.cpp wind.hpp
