FROM ghcr.io/prefix-dev/pixi:latest AS base

ARG CUDA_FLAVOR=cu128
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0"
ENV DEBIAN_FRONTEND=noninteractive
ENV PIXI_HOME=/root/.pixi
ENV WORKSPACE_ROOT=/opt/ember-core
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${WORKSPACE_ROOT}
COPY . ${WORKSPACE_ROOT}

CMD ["bash"]
