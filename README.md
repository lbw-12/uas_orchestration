# An Orchestration Engine for Scalable, On-Demand AI Phenotyping from UAS Imagery in Agriculture

## About
This repository contains the source code for the orchestration engine described in our ACM SIGSPATIAL 2025 paper:

An Orchestration Engine for Scalable, On-Demand AI Phenotyping from UAS Imagery in Agriculture > [Your Authors List] > 

Lucas Waltz
Department of Food, Agricultural,
and Biological Engineering
The Ohio State University
Columbus, Ohio, USA
waltz.12@osu.edu

Paul Rodriguez
San Diego Supercomputer Center,
University of California San Diego
San Diego, California, USA
prodriguez@sdsc.edu

Nicole DiMarco
Department of Computer Science and
Engineering
The Ohio State University
Columbus, Ohio, USA
dimarco.32@osu.edu

Sarikaa Sridhar
Department of Computer Science and
Engineering
The Ohio State University
Columbus, Ohio, USA
sridhar.86@osu.edu

Chaeun Hong
Department of Computer Science and
Engineering
The Ohio State University
Columbus, Ohio, USA
hong.930@osu.edu

Raghu Machiraju
Department of Computer Science and
Engineering
The Ohio State University
Columbus, Ohio, USA
machiraju.1@osu.edu

Ryan Waltz
Department of Food, Agricultural,
and Biological Engineering
The Ohio State University
Columbus, Ohio, USA
waltz.110@osu.edu

Armeen Ghoorkhanian
Department of Food, Agricultural,
and Biological Engineering
The Ohio State University
Columbus, Ohio, USA
ghoorkhanian.5@osu.edu

Sami Khanal
Department of Food, Agricultural,
and Biological Engineering
The Ohio State University
Columbus, Ohio, USA
khanal.3@osu.edu

Proceedings of the 33rd ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (SIGSPATIAL '25)

A live demonstration of the visualization outputs from this engine can be found at our project website: https://go.osu.edu/aerialagriviewer_sigspatial2025

## Overview
This project provides an open-source orchestration engine designed to automate and scale the transformation of raw Unmanned Aerial Systems (UAS) imagery into structured, AI-ready datasets for agricultural research. The pipeline is designed to operate in a High-Performance Computing (HPC) environment, leveraging a Slurm workload manager to handle large-scale data processing in an on-demand fashion.

The engine addresses the critical bottleneck of data processing for researchers, enabling them to move from raw aerial images to actionable, AI-driven phenotypic insights with minimal manual intervention.

*The data processing workflow of the orchestration engine, from raw imagery to visualization assets.*

## Key Features
- **Scalable Architecture**: Natively designed for HPC environments to process hundreds or thousands of flights in parallel
- **On-Demand Processing**: A watchdog-based folder watcher automatically initiates the processing pipeline as new imagery is uploaded
- **Modular Pipeline**: The workflow is composed of distinct, configurable modules that can be extended or modified for different use cases
- **Dual Georeferencing Pathways**: Supports both a conventional orthomosaic-based workflow and a higher-fidelity approach using direct georeferencing and image registration to preserve original image quality
- **AI-Ready Data Generation**: Enforces consistent Ground Sampling Distance (GSD) and performs temporal alignment to create datasets suitable for AI model training
- **Integrated AI Phenotyping**: Includes modules for inferring key agricultural traits such as crop growth stage, canopy cover, and spectral reflectance
- **Web-based Visualization**: Automatically generates a simple, web-based viewer for exploring imagery and the corresponding AI-inferred data overlays
- **Open Source**: The entire codebase is open source to foster collaboration and reproducibility in agricultural research

## Getting Started

### Prerequisites
- An HPC environment with a Slurm workload manager. The scripts are tailored for the environment at the Ohio Supercomputer Center (OSC) but can be adapted
- Conda for managing the Python environment. To recreate the environment:
  ```bash
  conda env create -f environment.yml
  conda activate harvest
  ```
- Singularity/Apptainer for running containerized software like OpenDroneMap

### Installation

1. Clone this repository: git clone https://github.com/lbw-12/uas_orchestration.git

### Setup

1. Generate yaml configuration file for your specific flight setup and workflow. Example files are located in this repo at yaml/
2. Create alignment points files for each field to align orthomosaics across time steps.
3. Create boundary files for flights that cover two or more fields.
4. Create plot shapefiles that correspond to unique treatments and ground truth observations.

### Testing
1. Initialize conda environment
 ```bash
 source ~/miniconda3/etc/profile.d/conda.sh 
 conda activate harvest
 ```
2. Start by running orchestrate.py using the orchestrate_onetime.sh script. Set flags to run through each step using the "--steps" argument and "--dry_run" argument to confirm each step is running properly on its own.
3. Continue by running orchestrate_onetime.sh across multiple steps working up to all steps to confirm all files are generating properly.

### Continuous Operation
1. Start folder_watcher program using orchestrate_ondemand.sh
 ```bash
 sbatch orchestrate_ondemand.sh
 ```