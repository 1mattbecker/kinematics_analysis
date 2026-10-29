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
    && rm -rf /var/lib/apt/lists/*

# ffmpeg 8.1 (GPL build, with libx264) in its own conda env so it cannot change the base env's
# packages, linked onto PATH. The apt ffmpeg (4.4 on Ubuntu 22.04) lacks -fps_mode, which the
# library's video_clips needs (ffmpeg >= 5.1, checked by video_clips.check_ffmpeg). Tested in this
# image's base on 2026-10-23. moviepy keeps its own bundled ffmpeg.
RUN mamba create -y -p /opt/ffmpeg -c conda-forge "ffmpeg=8.1.2=gpl*" \
    && mamba clean -afy \
    && ln -s /opt/ffmpeg/bin/ffmpeg /opt/ffmpeg/bin/ffprobe /usr/local/bin/ \
    && ffmpeg -hide_banner -version | head -1 \
    && ffmpeg -hide_banner -encoders | grep -q libx264

# Every pip install below uses -c /constraints.txt, which fixes the version of each package
# (see its header for history). Change a version there, not here, to upgrade it deliberately.
COPY constraints.txt /

# jupyterlab and ipywidgets: this base image has no IDE, and a hand-edited Dockerfile blocks
# Code Ocean's automatic IDE install. They move from 3.6 / 7.7 to current versions (JupyterLab
# 3.6 doesn't fit this base). No code in this repo uses ipywidgets directly.
# aind-dynamic-foraging-behavior-video-analysis is pinned to tag v0.4.0 (a6bac04): the video
# timing QC with timing_verdict, video screening, and frame-exact clips and frames
# (video_clips: cut_event_clips, write_frames). aind-video-utils is what its video-clips extra
# adds (scikit-learn, the other, is already here); it is listed directly rather than as
# [video-clips] on the editable install. To go back, pin tag v0.2.0 (0e1c8df; no video_clips)
# or v0.1.0 (41e5b59), or pre-video-timing-qc (5738b32) for the last version without the QC,
# and rebuild.
RUN pip install -U --no-cache-dir -c /constraints.txt \
    jupyterlab \
    ipywidgets \
    aind-ephys-utils==0.4.0 \
    hdmf-zarr==0.11.0 \
    fastparquet==2024.11.0 \
    pillow==10.3.0 \
    pyarrow==25.0.1 \
    pynwb==3.0.0 \
    scipy==1.18.1 \
    seaborn==0.13.2 \
    statsmodels==0.15.0 \
    zarr==2.18.2 \
    aind_analysis_arch_result_access --upgrade \
    -e git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-basic-analysis.git@main#egg=aind-dynamic-foraging-basic-analysis \
    -e git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-data-utils.git@main#egg=aind-dynamic-foraging-data-utils \
    -e git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-behavior-video-analysis.git@a6bac04805203413b876acf1d8b6509a28c66760#egg=aind-dynamic-foraging-behavior-video-analysis \
    aind-video-utils \
    opencv_python==4.11.0.86 \
    moviepy==1.0.3 \
    scikit-image==0.26.0 \
    trimesh==4.11.0
    # hdmf==3.12.2 \
    # aind_dynamic_foraging_data_utils==0.1.5

# rachel-analysis-utils requires Python >=3.10, which this image meets, so the
# --ignore-requires-python workaround needed on 3.9 is gone.
RUN pip install --no-cache-dir -c /constraints.txt \
    -e git+https://github.com/AllenNeuralDynamics/rachel-analysis-utils.git@864550d55356ecb4906d05801cdae694bd000fde#egg=rachel-analysis-utils

# scanpy, for eph_09_structural_axes' MERFISH axis (reads adata_mer_subset_2_2k.h5ad).
# Version set in the constraints file (1.12.4 since the 2026-09-28 package upgrade). scipy is
# re-asserted as a tripwire: if scanpy ever wants to move it, the build fails loudly instead of
# silently changing numpy/scipy under the rest of the pins.
RUN pip install --no-cache-dir -c /constraints.txt \
    scanpy \
    scipy==1.18.1

COPY postInstall /
RUN /postInstall
