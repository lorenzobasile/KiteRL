CXX = g++
CXXFLAGS = -O3 -std=c++14 -Ienv -Ienv/utils env/utils/utils.cpp
LIBFLAGS =  -Wno-return-type-c-linkage -shared -fpic  -o
OMPFLAGS = -fopenmp -static-libstdc++ -DPARALLEL
M1MAC = --target=x86_64-apple-darwin
#on M1 remove -fopenmp -static-libstdc++ and add --target=x86_64-apple-darwin
EXE = $(SRC:.cpp=.x)

.SUFFIXES:
SUFFIXES =

.SUFFIXES: .cpp .x

VPATH = env

all: libkite.so

parallel: libkite.cpp
			$(CXX) $(OMPFLAGS) $(LIBFLAGS) libkite.so $< $(CXXFLAGS)

m1: libkite.cpp
			$(CXX) $(M1MAC) $(LIBFLAGS) libkite.so $< $(CXXFLAGS)

x86: libkite.so


.PHONY: all


%.x: %.cpp

	$(CXX) $< -o $@ $(CXXFLAGS)

libkite.so: libkite.cpp
			$(CXX) $(LIBFLAGS) $@ $< $(CXXFLAGS)


clean:
		rm -f $(EXE) *~
		rm libkite.so

.PHONY: clean



libkite.so: kite.hpp constants.hpp vect.hpp env/utils/utils.cpp wind.hpp
