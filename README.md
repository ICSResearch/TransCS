# TransCS
This repo is the `pytorch` code for paper "TransCS: A Transformer-based Hybrid Architecture for Image Compressed Sensing".  
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
Before your test, please make sure you `have a GPU available` on your computer.  
Then, for example, if you want to test TransCS at sampling rate τ = 10%, please run :  
```
python eval.py --rate 0.1 --device 0
```  
For ease of use, this command will perform image sampling and reconstruction upon `all test datasets` at `one sampling rate`.
### 2. For re-training TransCS. 
* Put the `BSDS500 (.jpg)` folder (including training set, validation set and test set) into `./dataset/train`.  
* run 
* For example, if you want to train TranCS at τ = 10%, run `python train.py --rate --device 0`.
* Your re-trained models (.pth) will save in the `results folder`, it should contains `info.pth`, `model.pth`, `optimizer.pth` and `log.txt`, respectively represents the `result` in the training process (in order to start training from the breakpoint), `model parameters` and optimizer information, while log.txt saves the sampling and reconstruction `performance (PSNR, SSIM)` of the verification set after each training epoch.
****
## _Results_
Image reconstruction results of the butterfly and bird images by various methods in the case of sampling rates τ = 4%.
![image](https://github.com/myheuf/TransCS/blob/master/imgs/butterfly.png)
![image](https://github.com/myheuf/TransCS/blob/master/imgs/bird.png)
****
### END
For any questions, feel free to contact: shen65536@mail.nwpu.edu.cn
