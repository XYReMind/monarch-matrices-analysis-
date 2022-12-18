# 9143-HPML-Project 

Project Title: A Performance Analysis for Monarch Matrices.

## Introduction 

- To compare the performance of training Monarch matrix based transformer models with standard image classifiers.

## Monarch Matrices: 

- Naturally sparse
- Every matrix (transform) can be decomposed into Monarch matrices. (Prop 3.2, Dao et al.)
- Monarch: one of the first sparse training methods to achieve wall-clock speedup while
maintaining quality

![Alt text](/img/1.png?raw=true "sparse e2e")
![Alt text](/img/2.png?raw=true "ways to use sparse models")


## Environment and setup 
A guide to using Greene can be found here: https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/getting-started?authuser=0

You also need to connect to NYU VPN before using Greene. https://www.nyu.edu/life/information-technology/infrastructure/network-services/vpn.html is a guide to using VPN.

V100 GPU is used for this project. Mem is set to 8GB.


| Models          | System        | Hardware |Number of GPUS |Dataset      |Metrics                                       |
| ----------------|:-------------:|---------:|--------------:|------------:|---------------------------------------------:|
| ViT-16-S        | NYU HPC Greene| V100     | 4             | CIFAR-10    |Validation/Test Accuracy+Efficiency: wall-time|
| ViT-16-B        | NYU HPC Greene| V100     | 1             | CIFAR-10    |Validation/Test Accuracy+Efficiency: wall-time|
| MLP-mixer       | NYU HPC Greene| V100     | 1             | CIFAR-10    |Validation/Test Accuracy+Efficiency: wall-time|

## run  

1. ssh into NYU green
2. Using the btach file provided in this repo. (inside slurm folder)
3. Using command "squeue -u USERNAME" to check the progress of the training. 

## Result

| Models           | Cifar-10 Accuracy        | Speedup (Walltime) |Parameters    |FLOPS        |
| -----------------|:------------------------:|-------------------:|-------------:|------------:|
| ViT-16-S         | 68.01%                   | - (45.15376)       | 48.8M        | 9.9G        |
| ViT-16-S-Monarch | 78.16%                   | 1.8749(24.09827)   | 19.6M        | 3.9G        |
| ViT-16-B         | 78.6%                    | - (63.48)          | 89.6M        | 17.6G       |
| ViT-16-B-Monarch | 74.69%                   | -                  | 33.0M        | 5.91G       |
| MLP-mixer        | 78.9%                    | - (39.33)          | 59.9M        | 12.6G       |
| MLP-mixer-Monarch| 75.4%                    | -                  | 20.9M        | 5.0G        |


Reference:

[1]Dao et al, Monarch: Expressive Structured Matrices for Efficient and Accurate Training, ICML 2022, https://arxiv.org/pdf/2204.00595.pdf.
