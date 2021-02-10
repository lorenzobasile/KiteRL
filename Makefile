SRC = main.cpp
CXX = g++
CXXFLAGS = -std=c++14
EXE = $(SRC:.cpp=.x)

.SUFFIXES:
SUFFIXES =

.SUFFIXES: .cpp .x


all: $(EXE)


.PHONY: all

%.x: %.cpp

	$(CXX) $< -o $@ $(CXXFLAGS)



clean:
		rm -f $(EXE) *~

.PHONY: clean
