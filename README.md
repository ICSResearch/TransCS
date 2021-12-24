# _TransCS_
This repository is the `pytorch` code for our paper `"TransCS: A Transformer-based Hybrid Architecture for Image Compressed Sensing"`.  
We built and tested our code with an Intel Xeon Silver 4210 CPU and a GeForce RTX 3090 GPU.  
The complete code and data link will be put online soon.
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
Before your test, we recommend you `have a GPU available` on your computer, since a large number of CPU based tests are time-consuming.  
Then, for example, if you want to test TransCS at sampling rate τ = 10%, please run :  
```
python eval.py --rate 0.1 --device 0
```  
For ease of use, this command will perform image sampling and reconstruction upon `all test datasets` at `one sampling rate`.  
This is an example of the test results of dataset Set5 at sampling rate of 10% from the command line:  
```
Start evaluate...
Trained model loaded.
* ===================================  test dataset: ./dataset/test/SET5, device: cpu  =================================== *
=> process  1 done! time:  0.184s, PSNR: 35.07, SSIM: 0.9308, name: ./dataset/test/SET5/(1).mat
=> process  2 done! time:  0.152s, PSNR: 37.53, SSIM: 0.9684, name: ./dataset/test/SET5/(2).mat
=> process  3 done! time:  0.156s, PSNR: 29.72, SSIM: 0.9317, name: ./dataset/test/SET5/(3).mat
=> process  4 done! time:  0.150s, PSNR: 33.32, SSIM: 0.8447, name: ./dataset/test/SET5/(4).mat
=> process  5 done! time:  0.159s, PSNR: 32.16, SSIM: 0.9464, name: ./dataset/test/SET5/(5).mat
=> All the  5 images done!, your AVG PSNR: 33.56, AVG SSIM: 0.9244
```
### 3. For re-training TransCS. 
* Put the `BSDS500 (.jpg)` folder (including training set, validation set and test set) into `./dataset/train`.  
* For example, if you want to train TranCS at τ = 10%, please run the following command. The BSDS500 will be automatically packaged and trained with default parameters (please ensure 24G video memory or more).
```
python train.py --rate --device 0
```
* Your re-trained models (.pth) will save in the `results folder`, it should contains `info.pth`, `model.pth`, `optimizer.pth` and `log.txt`, respectively represents the `result` in the training process (in order to start training from the breakpoint), `model parameters` and optimizer information, while log.txt saves the sampling and reconstruction `performance (PSNR, SSIM)` of the verification set after each training epoch.  
****
## _Examples of Results_
Partial visual comparisons of the *`butterfly`* and *`bird`* (from dataset Set5) reconstruction images by multiple methods at sampling rates τ = 4%.  
Please refer to our paper for more results and comparisions.  
![image](https://github.com/myheuf/TransCS/blob/master/imgs/butterfly.png)
![image](https://github.com/myheuf/TransCS/blob/master/imgs/bird.png)
****
### END
For any questions, feel free to contact us: shen65536@mail.nwpu.edu.cn
