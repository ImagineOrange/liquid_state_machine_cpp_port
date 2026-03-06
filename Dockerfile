FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    make \
    zlib1g-dev \
    liblapack-dev \
    libblas-dev \
    libgomp1 \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir numpy matplotlib scikit-learn scipy

WORKDIR /workspace

COPY . /workspace/

# Build the C++ binary
RUN make clean 2>/dev/null; make CXX=g++ -j$(nproc)

CMD ["/bin/bash"]
