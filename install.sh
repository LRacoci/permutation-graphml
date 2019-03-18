#!/bin/bash

##### MODE SELECTION

# Select adequate mode
read -p "What mode would you to install in (cpu/gpu)?" mode

# Check the selected mode exists
if [ mode -eq "cpu" ] || [ mode -eq "gpu" ]; then
    # Log message
    echo "Installing as $mode..."
fi
else
    # Exit w/ error status
    exit -1;
fi

##### INSTALL
# If cpu, install using cpu mode
if [ mode -eq "cpu" ]; then
    # Pull image with dependencies from floydhub
    docker pull floydhub/dl-docker:cpu
    # Install graph_nets library
    docker run --name dl-docker dl-docker bash -c "pip install graph_nets"
    # Commit (save) changes to image
    docker commit dl-docker graph-nets-docker
fi
# Else, install using gpu mode
else
    # Check user has configured gpu steps first
    read -p "Have you configured the gpu drivers and nvidia-cli in \"configure-gpu\" (Y/N)?" ans
    if [ ans -neq "Y" ]; then
        echo "Configure gpu first in \"configure-gpu\""
        exit -1;
    fi
    else
        # Clone repo and move to folder
        git clone https://github.com/saiprashanths/dl-docker.git
        cd dl-docker
        # Build gpu image from floyhub 
        docker build -t floydhub/dl-docker:gpu -f Dockerfile.gpu . 
        # Install graph_nets library
        docker run --name dl-docker dl-docker bash -c "pip install graph_nets"
        # Commit (save) changes to image
        docker commit dl-docker graph-nets-docker
    fi
    
fi
