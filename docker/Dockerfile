FROM debian:trixie-slim

ENV DEBIAN_FRONTEND=noninteractive \
    MAKEFLAGS="-j$(nproc)"

# Install build tools + minimal runtime libs (no recommends)
RUN apt-get update && apt install -y --no-install-recommends \
    build-essential \
    g++ \
    make \
    zlib1g-dev \
    liblapack-dev \
    libblas-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt -y autoremove 

# Set up workspace
WORKDIR /workspace

# Create a non-root user (good practice)
RUN useradd -m -u 1000 user && chown -R user /workspace
USER user

# Default shell
CMD ["/bin/bash"]
