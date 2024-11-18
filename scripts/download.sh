#!/bin/bash

# Downloads the Checkpoints related to the Backbone
mkdir weights

# ViT-B/32 https://github.com/marqo-ai/GCL
# Source: https://marqo-gcl-public.s3.us-west-2.amazonaws.com/v1/marqo-gcl-vitb32-127-gs-full_states.pt
echo "[INFO] Downloading the Pre-Trained Backbone for ViT-B/32"
wget https://marqo-gcl-public.s3.us-west-2.amazonaws.com/v1/marqo-gcl-vitb32-127-gs-full_states.pt -O ./weights/marqo-gcl-vitb32-127-gs-full_states.pt

# ViT-L/14 https://github.com/marqo-ai/GCL
# Source: https://marqo-gcl-public.s3.us-west-2.amazonaws.com/v1/marqo-gcl-vitl14-124-gs-full_states.pt
echo "[INFO] Downloading the Pre-Trained Backbone for ViT-L/14"
wget https://marqo-gcl-public.s3.us-west-2.amazonaws.com/v1/marqo-gcl-vitl14-124-gs-full_states.pt -O ./weights/marqo-gcl-vitl14-124-gs-full_states.pt