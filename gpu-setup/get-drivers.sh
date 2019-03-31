#!/bin/bash

# Add repository
sudo add-apt-repository ppa:graphics-drivers/ppa
# Update list of repos
sudo apt-get update
# Install driver version recommended according to:
# https://github.com/floydhub/dl-setup#nvidia-drivers
sudo apt-get install nvidia-352

# Restart
sudo shutdown -r now