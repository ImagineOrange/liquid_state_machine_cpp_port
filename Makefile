CXX      ?= g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -Wno-unused-parameter \
           -Wno-deprecated-declarations -Wno-unused-function

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LDFLAGS += -framework Accelerate
    LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
    ifneq ($(LIBOMP_PREFIX),)
        CXXFLAGS += -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
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

# Source files
SRCS = $(SRCDIR)/main.cpp \
       $(SRCDIR)/ml.cpp \
       $(SRCDIR)/network.cpp \
       $(SRCDIR)/builder.cpp \
       $(SRCDIR)/npz_reader.cpp

OBJS = $(SRCS:.cpp=.o)

TARGET = cls_sweep

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile rule: include headers from `inc/`
$(SRCDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)
