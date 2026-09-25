# hash:sha256:2e986011032f62d32c9f0c8a6bdec40fd30fdc1642dfd1e39cfe52a608f98f19
ARG REGISTRY_HOST
# AIND capsule template base: Python 3.12.4 (mamba/conda-forge), Ubuntu 22.04.
# Replaces jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04. See
# PYTHON_311_UPGRADE_PLAN.md in aind-dynamic-foraging-behavior-video-analysis, Stage 2.
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-ask-pass /

RUN python -m pip install --upgrade pip

# Versions are no longer pinned to Ubuntu 20.04 builds. python3-tk is gone: it was the
# system Python's Tk, and conda-forge's Python ships its own. git and curl are listed
# explicitly because this base image is minimal: git for the pip installs from GitHub,
# curl for postInstall's code-server download.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libgit2-dev \
        libgl1-mesa-glx \
        pandoc \
        pkg-config \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Every pip install below uses -c /py39-constraints.txt, which holds each package at its
# version from the Python 3.9 baseline (environment/py39-freeze.txt). Change a version there,
# not here, to upgrade it deliberately.
COPY py39-constraints.txt /

# The three AIND libraries are pinned to the commits in the 3.9 baseline, not @main, so the
# reference comparison isn't muddied by upstream changes made since (basic-analysis has
# touched metrics/trial_metrics.py). TEMPORARY: set them back to @main when adopting (plan 2c).
#
# jupyterlab and ipywidgets: this base image has no IDE, and a hand-edited Dockerfile blocks
# Code Ocean's automatic IDE install. They move from 3.6 / 7.7 to current versions (JupyterLab
# 3.6 doesn't fit this base). No code in this repo uses ipywidgets directly.
RUN pip install -U --no-cache-dir -c /py39-constraints.txt \
    jupyterlab \
    ipywidgets \
    aind-ephys-utils==0.0.15 \
    hdmf-zarr==0.11.0 \
    fastparquet==2024.11.0 \
    open-ephys-python-tools==0.1.7 \
    pillow==10.3.0 \
    pyarrow==21.0.0 \
    pymupdf==1.24.2 \
    pynwb==3.0.0 \
    pypdf2==3.0.1 \
    scipy==1.13.0 \
    seaborn==0.13.2 \
    spikeinterface[full]==0.100.0 \
    statsmodels==0.14.2 \
    wavpack-numcodecs==0.1.5 \
    zarr==2.18.2 \
    aind_analysis_arch_result_access --upgrade \
    -e git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-basic-analysis.git@4dccc525947241fd5278927de5cf1ad7fe800e4d#egg=aind-dynamic-foraging-basic-analysis \
    -e git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-data-utils.git@f04420e073b280579ff822c9754c32e453710043#egg=aind-dynamic-foraging-data-utils \
    -e git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-behavior-video-analysis.git@de27558a96a1104757082a565c6b688fd4d0dcff#egg=aind-dynamic-foraging-behavior-video-analysis \
    opencv_python==4.11.0.86 \
    moviepy==1.0.3 \
    scikit-image==0.24.0 \
    trimesh==4.11.0
    # hdmf==3.12.2 \
    # aind_dynamic_foraging_data_utils==0.1.5

# rachel-analysis-utils requires Python >=3.10, which this image meets, so the
# --ignore-requires-python workaround needed on 3.9 is gone.
RUN pip install --no-cache-dir -c /py39-constraints.txt \
    -e git+https://github.com/AllenNeuralDynamics/rachel-analysis-utils.git@864550d55356ecb4906d05801cdae694bd000fde#egg=rachel-analysis-utils

# scanpy, for eph_09_structural_axes' MERFISH axis (reads adata_mer_subset_2_2k.h5ad).
# Held at 1.10.3 (the 3.9 baseline) by the constraints file, so the migration changes as little
# as possible. It was pinned there only because 1.10.4+ need Python >=3.10, so it can be
# bumped now. scipy is re-asserted as a tripwire: if scanpy ever wants to move it, the build
# fails loudly instead of silently changing numpy/scipy under the rest of the pins.
RUN pip install --no-cache-dir -c /py39-constraints.txt \
    scanpy \
    scipy==1.13.0

COPY postInstall /
RUN /postInstall
