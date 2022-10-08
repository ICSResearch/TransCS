# TransCS
TBD.

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
An example of running results are as follows:
```
Start evaluate...
Trained model loaded.
* ============  test dataset: ./dataset/test/SET5, device: cuda:0 ============= *
=> process  1 done! PSNR: 35.07, SSIM: 0.9308, name: ./dataset/test/SET5/(1).mat
=> process  2 done! PSNR: 37.53, SSIM: 0.9684, name: ./dataset/test/SET5/(2).mat
=> process  3 done! PSNR: 29.72, SSIM: 0.9317, name: ./dataset/test/SET5/(3).mat
=> process  4 done! PSNR: 33.32, SSIM: 0.8447, name: ./dataset/test/SET5/(4).mat
=> process  5 done! PSNR: 32.16, SSIM: 0.9464, name: ./dataset/test/SET5/(5).mat
=> All the  5 images done!, your AVG PSNR: 33.56, AVG SSIM: 0.9244
```
### 3. For re-training TransCS. 
Put the `BSDS500 (.jpg)` folder (including training set, validation set and test set) into `./dataset/train`.  
For example, if you want to train TranCS at τ = 10%, please run the following command. The BSDS500 will be automatically packaged and our TransCS will be trained with default parameters (please ensure 24G memory or more).
```
python train.py --rate 0.1 --device 0
```
Your re-trained models (.pth) will be saved in the `results folder`, it should contains `info.pth`, `model.pth`, `optimizer.pth` and `log.txt`, respectively represents the `result` in the training process (in order to start training from the breakpoint), `model parameters` and optimizer information, while log.txt saves the sampling and reconstruction `performance (PSNR, SSIM)` of the verification after each training epoch.  

## Examples of Results
Partial visual comparisons of reconstructed images by multiple methods at sampling rates τ = 0.04 and 0.25.  
Please refer to our paper for more results and comparisions.  
<div align=center><img width="600" height="350" src="https://github.com/myheuf/TransCS/blob/master/imgs/demo.png"/></div>
