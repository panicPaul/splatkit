FROM ghcr.io/prefix-dev/pixi:latest AS build

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

RUN pixi install --frozen -e ${CUDA_FLAVOR}
RUN pixi run -e ${CUDA_FLAVOR} uv sync --locked --extra ${CUDA_FLAVOR}
RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'export PIXI_ENV_PREFIX=/opt/ember-core/.pixi/envs/'"${CUDA_FLAVOR}" \
    'export CUDA_HOME="${PIXI_ENV_PREFIX}"' \
    'export CUDA_PATH="${PIXI_ENV_PREFIX}"' \
    'export PATH="${PIXI_ENV_PREFIX}/bin:${PATH}"' \
    'exec "$@"' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

FROM ubuntu:24.04 AS runtime

ARG CUDA_FLAVOR=cu128
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0"
ENV WORKSPACE_ROOT=/opt/ember-core
ENV PIXI_ENV_PREFIX=/opt/ember-core/.pixi/envs/${CUDA_FLAVOR}
ENV CUDA_HOME=${PIXI_ENV_PREFIX}
ENV CUDA_PATH=${PIXI_ENV_PREFIX}
ENV PATH=${PIXI_ENV_PREFIX}/bin:${PATH}
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates bash git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${WORKSPACE_ROOT}
COPY --from=build /opt/ember-core /opt/ember-core
COPY --from=build /entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-c", "import torch; print(torch.__version__)"]
