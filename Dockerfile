# Start from an official NVIDIA CUDA base image
FROM nvidia/cuda:12.6.0-base-ubuntu22.04

# Set up the working directory
WORKDIR /app

# Install Miniconda

# Install necessary system packages like curl AND apptainer in a single step
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    apptainer \
    && rm -rf /var/lib/apt/lists/*

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Set Conda environment variables to handle ToS and prevent auto-updates
ENV CONDA_AUTO_UPDATE_CONDA=false

# Accept Anaconda ToS for the required channels
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Copy your local files into the container
COPY . .

# Copy the Apptainer image for ODM to the container
RUN gsutil cp gs://data-uas/containers/odm_latest.sif /app/execution/containers/

# Create the environment from your file in a single, non-interactive step
RUN conda env create -f environment_docker.yml

# Set up the shell to use the Conda environment by default
SHELL ["conda", "run", "-n", "harvest", "/bin/bash", "-c"]