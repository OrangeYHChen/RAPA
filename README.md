# RAPA

### Prerequisites
* Pytorch 1.1
* cuda 9.0
* python 3.6
* GPU Memory>20G We Recommend Titan RTX or Tesla V100

### Datasets
We evaluate our method on Mars, iLIDS-VID and PRID-2011 datasets. You can download datasets from [Here](https://kaiyangzhou.github.io/deep-person-reid/datasets.html#video-datasets), and put them into /data/datasets/.

### Usage
* Firstly, we provide the region box information which extracts from Mars，iLIDS-VID and PRID-2011 datasets with the application of HRNet. You can download from the following links and put them into /data/keypoints/.

[MARS_Testing_RegionBox](https://drive.google.com/file/d/1OTcEfFHUI-nkMU8l5ZGqN4hDol6exmgD/view?usp=sharing)|
[MARS_Training_RegionBox](https://drive.google.com/file/d/1wk-P7fTW7sJpWLMmqlYkgJGT2X9H2fWP/view?usp=sharing)|
[iLIDS-VID Testing and Training RegionBox](https://drive.google.com/file/d/1Q8G6MUCCIMK21mFNfmBz4gl6qKd-ZhfN/view?usp=sharing)|
[PRID-2011 Testing and Training RegionBox](https://drive.google.com/file/d/1mGIFNPaGsMRHjCd5dJbmwauqzjKWRdAE/view?usp=sharing)

* If you want to test our trained model on MARS, you can obtain our trained model from [Here](https://drive.google.com/file/d/1qpJKPgPLyHriiNfBoJGRDbVGcOqAxhBo/view?usp=sharing), and put it into /weights/. After that, you can run our code with the command `` python evaluate.py``.
* If you want to train the network, you can run our code with the following commands:

On Mars dataset: 

``
python run.py --max_epoch 400 --train_batch 32 --lr 0.00035 --feat_dim 1024 --a1 1 --a2 1 --a3 0.0003 --gpu_devices 0
``

On iLIDS-VID dataset: 

``
python run.py --max_epoch 400 --train_batch 32 --lr 0.00035 --feat_dim 512 --a1 1 --a2 1 --a3 0.00005 --gpu_devices 0
``

On PRID-2011 dataset: 

``
python run.py --max_epoch 400 --train_batch 32 --lr 0.00035 --feat_dim 256 --a1 1 --a2 1 --a3 0.00005 --gpu_devices 0
``

### Evaluate
| Dataset | Rank1 | Rank5 | Rank20 |mAP|
| :------: | :------: | :------: | :------: | :------: |
| MARS | 88.7 | 96.1 | 98.1 | 82.8 |
| PRID2011 | 95.2 | 99.2 | 100.0 | - |
| iLIDS-VID | 89.6 | 98.0 | 99.9 | - |
