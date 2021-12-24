# TransCS
This repo is the pytorch code for paper "TransCS: A Transformer-based Hybrid Architecture for Image Compressed Sensing".  
We built and tested with an Intel Xeon Silver 4210 CPU and a GeForce RTX 3090 GPU.  
The complete code will be put online soon.
****
## _Requirements_
Python 3.9  
CUDA 11.1  
PyTorch 1.9.0  
torchvision 0.9.0  
****
## _Useage_
1. File directory description:  
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
****
## _Results_
Image reconstruction results of the butterfly and bird images by various methods in the case of sampling rates τ = 4%.
![image](https://github.com/myheuf/TransCS/blob/master/imgs/butterfly.png)
![image](https://github.com/myheuf/TransCS/blob/master/imgs/bird.png)
