# RAPA

### Prerequisites
* Pytorch 1.0
* cuda 9.0
* python 3.6
* GPU Memory>20G We Recommend Titan RTX or Tesla V100

### Datasets
To use our code, you should download MARS dataset from [Here](https://pan.baidu.com/s/1XKBdY8437O79FnjWvkjusw)(key:ymc5), and put these data into /data/datasets/mars/.

### Usage
* Firstly, we provide the region box information which extracts from MARS dataset with the application of HRNet. You can download from [MARS_Test_RegionBox](https://drive.google.com/file/d/1OTcEfFHUI-nkMU8l5ZGqN4hDol6exmgD/view?usp=sharing) and [MARS_Train_RegionBox](https://drive.google.com/file/d/1wk-P7fTW7sJpWLMmqlYkgJGT2X9H2fWP/view?usp=sharing), and put them into /data/keypoints/.
* If you want to test our trained model, you can obtain our trained model from [Here](https://drive.google.com/file/d/1qpJKPgPLyHriiNfBoJGRDbVGcOqAxhBo/view?usp=sharing), and put it into /weights/. After that, you can run our code with the command "python evaluate.py".
* If you want to train the network, you can run our code with the command "python run.py".

### Evaluate
| Dataset | Rank1 | Rank5 | Rank20 |mAP|
| :------: | :------: | :------: | :------: | :------: |
| MARS | 88.7 | 96.1 | 98.1 | 82.8 |

 
