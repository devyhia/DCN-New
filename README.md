# Deep Clustering Network (DCN)

This work is a modified codebase of: https://github.com/devyhia/DCN-New

### What is different?
1. We added a Dockerfile that contains the environment for which this trains and evaluates.
2. We added the ability to train the COIL-20 dataset.
3. We fine-tuned the hyperparameters for the network to obtain the best possible performance on COIL-20 dataset using this network.

### How to train the network?

1. Install Docker with Nvidia GPU support. This could be done following this tutorial: https://github.com/NVIDIA/nvidia-docker
2. Build the docker image using the following snippet:
```
docker build -t dcn .
```
3. Open the docker image (i.e. bash into it).
```
docker run --runtime=nvidia --rm dcn bash
```
3. Inside the bash, start training the network using the following snippet.
```
THEANO_FLAGS='floatX=float32,device=cuda,dnn.enabled=False' python run_raw_coil20.py
```
