#!/bin/bash

# Constants
CPU_IMG="floydhub/dl-docker:cpu"
CPU_CTN="floydhub-dl-docker-cpu"

# Pull image with dependencies from floydhub
sudo docker pull $CPU_IMG
# Install graph_nets library
sudo docker run --name $CPU_CTN $CPU_IMG bash -c "pip install graph_nets"
# Copy project files
sudo docker cp . $CPU_CTN:/"\$HOME"
# Commit changes from container
sudo docker commit $CPU_CTN $CPU_IMG