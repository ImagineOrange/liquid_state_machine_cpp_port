CXX      ?= g++
CXXFLAGS  = -std=c++17 -O3 -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-declarations -Wno-unused-function
LDFLAGS   =

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LDFLAGS += -framework Accelerate
    # OpenMP via Homebrew libomp (install: brew install libomp)
    LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
    LIBOMP_INC := $(wildcard $(LIBOMP_PREFIX)/include/omp.h)
    ifneq ($(LIBOMP_INC),)
        CXXFLAGS += -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
        LDFLAGS  += -L$(LIBOMP_PREFIX)/lib -lomp
    endif
else
    # Linux (Ubuntu, etc.)
    CXXFLAGS += -fopenmp
    LDFLAGS  += -llapack -lblas -lgomp
endif

LDFLAGS += -lz -lpthread

SRCDIR = cpp
SRCS   = $(SRCDIR)/npz_reader.cpp \
         $(SRCDIR)/network.cpp \
         $(SRCDIR)/builder.cpp \
         $(SRCDIR)/ml.cpp \
         $(SRCDIR)/main.cpp
OBJS   = $(SRCS:.cpp=.o)
TARGET = cls_sweep

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(SRCDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)
