#!/bin/bash

# Constants
GPU_IMG="floydhub/dl-docker:gpu"
GPU_CTN="floydhub-dl-docker-gpu"

# GPU pre-configured?
echo "Have you run \"gpu_part1.sh\" and \"gpu_part2.sh\" (Y/n)?"
read ans
# If not => exit
if [ "$ans" != "Y" ]; then
    echo "Configure gpu first in \"configure-gpu\""
    exit -1;
# If yes, proceed with rest of setup
else
    # Clone repo and move to folder
    git clone https://github.com/saiprashanths/dl-docker.git
    cd dl-docker
    # Build gpu image from floyhub 
    sudo docker build -t $GPU_IMG -f Dockerfile.gpu . 
    # Install graph_nets library
    sudo docker run --name $GPU_CTN $GPU_IMG bash -c "pip install graph_nets"
    # Copy project files
    sudo docker cp . $GPU_CTN:/"\$HOME"
    # Commit changes from container
    sudo docker commit $GPU_CTN $GPU_IMG
fi
