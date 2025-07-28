# hash:sha256:2e986011032f62d32c9f0c8a6bdec40fd30fdc1642dfd1e39cfe52a608f98f19
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04
 
ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY
 
ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /
 
RUN python -m pip install --upgrade pip

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.8ubuntu1.1 \
        libgit2-dev=0.28.4+dfsg.1-2ubuntu0.1 \
#        libssl-dev=1.1.1f-1ubuntu2.22 \
        libgl1-mesa-glx \
        pandoc=2.5-3build2 \
        pkg-config=0.29.1-0ubuntu4 \
        python3-tk=3.8.10-0ubuntu1~20.04 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-ephys-utils==0.0.15 \
    hdmf-zarr==0.11.0 \
    ipywidgets==7.7.2 \
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
    -e git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-basic-analysis.git@main#egg=aind-dynamic-foraging-basic-analysis \
    -e git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-data-utils.git@main#egg=aind-dynamic-foraging-data-utils \
    aind_analysis_arch_result_access \
    -e git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-behavior-video-analysis.git@main#egg=aind-dynamic-foraging-behavior-video-analysis

    # opencv_python==4.11.0.86 \
    # moviepy==1.0.3 \
    # hdmf==3.12.2 \
    # aind_dynamic_foraging_data_utils==0.1.5

COPY postInstall /
RUN /postInstall

