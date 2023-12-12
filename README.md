# MLNet
[AAAI2024] MLNet: Mutual Learning Network with Neighborhood Invariance for Universal Domain Adaptation

## Installation

Please refer to the following official websites or download links to get datasets.

| Dataset | Official Website | Download Link |
| ------- | ---------------- | ------------- |
| Office | [https://faculty.cc.gatech.edu/~judy/domainadapt/](https://faculty.cc.gatech.edu/~judy/domainadapt/) | [https://drive.google.com/open?id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE](https://drive.google.com/open?id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE) |
| OfficeHome | [https://www.hemanthdv.org/officeHomeDataset.html](https://www.hemanthdv.org/officeHomeDataset.html) | [https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw) |
| VisDA | [http://ai.bu.edu/visda-2017/](http://ai.bu.edu/visda-2017/) | [https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) |

Then create corresponding symbol links in the `data/` folder as follows. Please make sure that the name of each dataset folder is the same as given below, otherwise the path in the image list may become invalid. Don't forget to create an extra symbol link for the OfficeHome dataset to remove the space in the Read-world domain.
```
ln -s PATH_TO_YOUR_OFFICE data/Office
ln -s PATH_TO_YOUR_OFFICEHOME data/OfficeHome
ln -s PATH_TO_YOUR_VISDA data/VisDA
ln -s data/OfficeHome/Real\ World data/OfficeHome/Real
```

The directory structure of `data/` should be as follows.
```
data/
├── Office -> /home/xxx/datasets/office
│   ├── amazon
│   ├── dslr
│   └── webcam
├── OfficeHome -> /home/xxx/datasets/OfficeHomeDataset_10072016
│   ├── Art
│   ├── Clipart
│   ├── ImageInfo.csv
│   ├── imagelist.txt
│   ├── Product
│   ├── Real -> Real World
│   └── Real World
├── splits
│   ├── Office
│   ├── OfficeHome
│   └── VisDA
└── VisDA -> /home/xxx/datasets/VisDA
    ├── image_list.txt
    ├── test
    ├── train
    └── validation
```

Finally, create the anaconda environment to complete the installation.
```
conda create -f environment.yaml
```

## Run

We have provided a lot of running scripts in the `scripts/` folder for single or multiple runs. For example, to perform UniDA training from Amazon(A) to DSLR(D) in the CDA setting on the Office dataset:
```
bash scripts/MLNet/Office/CDA/amazon2dslr.sh GPU_ID
```
If you want to modify some preset hyperparameters or settings in the ``configs/`` folder, like doubling the learning rate for backbone:
```
bash scripts/MLNet/Office/CDA/amazon2dslr.sh GPU_ID OPTIMIZER.FT_LR 0.002
```
It is also possible to train a single setting or all settings of the same dataset at once:
```
bash scripts/MLNet/Office/train_all_cda.sh GPU_ID
bash scripts/MLNet/Office/train_all.sh GPU_ID
```
Note that you don't need to activate the environment or set the number of threads by yourself, it's all automatic. When training MLNet on the VisDA dataset, the required memory can be large. The above scripts are also available for the baseline OVANet.


## Citation
```
@inproceedings{lu2024mlnet,
  title={MLNet: Mutual Learning Network with Neighborhood Invariance for Universal Domain Adaptation},
  author={Lu, Yanzuo and Shen, Meng and Ma, Andy J and Xie, Xiaohua and Lai, Jian-Huang},
  booktitle={AAAI},
  year={2024}
}
```