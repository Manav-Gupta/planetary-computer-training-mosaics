# Bakes the halo-s2 conda env (incl. omnicloudmask + CPU torch) into the
# image at build time, instead of installing it fresh on every job run.
# Build context is the repo root (see azureml/environment/environment.yml).
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04

COPY environment.yml /tmp/environment.yml

RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# AML's MCR base images keep miniconda at /opt/miniconda; this variable is
# how AML's job runtime finds and activates a non-default conda env baked
# into a custom image. If a job can't find the env after a rebuild, check
# this path still matches (`conda env list` inside the image).
ENV AZUREML_CONDA_ENVIRONMENT_PATH=/opt/miniconda/envs/halo-s2
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH
