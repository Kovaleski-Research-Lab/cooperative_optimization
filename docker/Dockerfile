#Use NVIDIA's PyTorch base image
FROM nvcr.io/nvidia/pytorch:22.06-py3
 
# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    tmux \
    vim \
    htop \
    git \
    wget
 
RUN pip install --no-cache-dir matplotlib tqdm pandas wandb pillow
WORKDIR /develop/build
RUN git clone https://github.com/Kovaleski-Research-Lab/diffractive_optical_model.git
RUN pip install -e diffractive_optical_model/
RUN pip install seaborn
RUN pip install umap-learn

WORKDIR /develop/code
RUN git clone https://github.com/Kovaleski-Research-Lab/cooperative_optimization.git
 
# Set the working directory
WORKDIR /develop/code
WORKDIR /develop/results
WORKDIR /develop/data
 
