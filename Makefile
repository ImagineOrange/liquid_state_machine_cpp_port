# Compiler and flags
CXX      ?= g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -Wno-unused-parameter \
	-Wno-deprecated-declarations -Wno-unused-function
LDFLAGS  =

# Platform-specific flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	CXXFLAGS += -Xpreprocessor -fopenmp
	LDFLAGS  += -framework Accelerate
	LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
	ifneq ($(LIBOMP_PREFIX),)
	CXXFLAGS += -I$(LIBOMP_PREFIX)/include
	LDFLAGS  += -L$(LIBOMP_PREFIX)/lib -lomp
	endif
else
	CXXFLAGS += -fopenmp
	LDFLAGS  += -llapack -lblas -lgomp
endif

LDFLAGS += -lz -lpthread

# Directory layout
SRCDIR = src/src
INCDIR = src/inc
TESTDIR = src/tests

# Source files
SRCS = $(SRCDIR)/main.cpp \
	$(SRCDIR)/ml.cpp \
	$(SRCDIR)/network.cpp \
	$(SRCDIR)/builder.cpp \
	$(SRCDIR)/npz_reader.cpp

# Object files for main build
OBJS = $(SRCS:.cpp=.o)

TARGET = cls_sweep

# Test-related
TEST_TARGET = cls_tests
TEST_SOURCES = $(TESTDIR)/test_suite.cpp \
	$(wildcard $(SRCDIR)/*.cpp)
TEST_HEADERS = $(wildcard $(INCDIR)/*.h)

.PHONY: all clean test build-tests

all: $(TARGET)

# Main binary build
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	rm -f $(OBJS)

# Compile main source files
$(SRCDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c -o $@ $<

# Build tests using CMake
build-tests: $(TEST_TARGET)

$(TEST_TARGET):
	@mkdir -p build-tests
	@cmake -S . -B build-tests -DCMAKE_BUILD_TYPE=Release
	@cmake --build build-tests

# Run tests
test: build-tests
	@build-tests/$(TEST_TARGET)

# Clean everything
clean:
	rm -f $(OBJS) $(TARGET)
	rm -rf build-tests
