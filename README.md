# TransCS
This repository is the `pytorch` code for our paper `"TransCS: A Transformer-based Hybrid Architecture for Image Compressed Sensing"`.  
We built and tested our code with an Intel Xeon Silver 4210 CPU and a GeForce RTX 3090 GPU.  

## Datasets and Environments
Training dataset: `BSDS500`, validation dataset: `Set11`, testing datasets: `Set5`, `BSD100` and `Urban100`.  
Runtime environment: `Python 3.9`, `CUDA 11.1`, `PyTorch 1.9.0`, `torchvision 0.9.0`.  

## Useage
### 1. Directory description:  
```
TransCS (Project name)  
|-dataset
|    |-train  
|        |-BSDS500 (.jpg)  
|    |-val  
|        |-Set11 (.mat)  
|    |-test  
|        |-Set5 (.mat)  
|        |-BSDS100 (.mat)  
|        |-Urban100 (.mat)  
|-models
|    |-__init__.py  
|    |-demo.py  
|    |-module.py  
|-results  
|    |-4  
|    |-10  
|    |-25  
|    |-... (sampling rates)
|-utils 
|    |-__init__.py  
|    |-config.py  
|    |-loader.py  
|-eval.py  
|-train.py
```
### 2. For testing the trained network of TransCS.  
Before your testing, we recommend you `have a GPU available`, since a large number of CPU based tests are very time-consuming.  
Then, for example, if you want to test TransCS at sampling rate τ = 0.10, please run :  
```
python eval.py --rate 0.1 --device 0
```  
For ease of use, this command will perform image sampling and reconstruction upon `all test datasets` at `one sampling rate`.  
### 3. For re-training TransCS. 
Put the `BSDS500 (.jpg)` folder (including training set, validation set and test set) into `./dataset/train`.  
For example, if you want to train TranCS at τ = 10%, please run the following command. The BSDS500 will be automatically packaged and our TransCS will be trained with default parameters (please ensure 24G memory or more).
```
python train.py --rate 0.1 --device 0
```
Your re-trained models (.pth) will be saved in the `results folder`, it should contains `info.pth`, `model.pth`, `optimizer.pth` and `log.txt`, respectively represents the `result` in the training process (in order to start training from the breakpoint), `model parameters` and optimizer information, while log.txt saves the sampling and reconstruction `performance (PSNR, SSIM)` of the verification after each training epoch.  

## _Examples of Results_
Partial visual comparisons of reconstructed images by multiple methods at sampling rates τ = 0.04 and 0.25.  
Please refer to our paper for more results and comparisions.  
<div align=center><img width="600" height="350" src="https://github.com/myheuf/TransCS/blob/master/imgs/demo.png"/></div>  

### _END_
For any questions, feel free to contact us: shen65536@mail.nwpu.edu.cn
