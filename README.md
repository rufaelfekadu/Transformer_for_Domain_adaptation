# Transformer for Unsupervised Domain Adaptation

## Introduction
This repo containes the implementation of a crossdomain transformer architecture adopted from [CDTrans](https://github.com/CDTrans/CDTrans.git) for domain adaptation in Urban Road Scene Perception

## Results
#### Table 1 Accuracy values for the model trained on the Source domain and tested on the target domain(Source only), model trained and evaluated on the target domain(Target only) and UDA. The UDA per-class accuracy is also shown.
<table>
    <tr>
        <td>Method Name</td>
        <td>Source Only</td>
        <td>Target Only</td>
        <td>Bicycle</td>
        <td>Car</td>
        <td>Person</td>
        <td>UDA</td>
    </tr>
    <tr>
        <td>DeiT-small</td>
        <td>94.20</td>
        <td>97.50</td>
        <td>73.00</td>
        <td>extbf{98.00}</td>
        <td>95.00</td>
        <td>95.90</td>
    </tr>
    <tr>
        <td>DeiT-base</td>
        <td>\textbf{94.80}</td>
        <td>97.60</td>
        <td>77.00</td>
        <td>97.00</td>
        <td>97.00</td>
        <td>\textbf{96.40}</td>
    </tr>
    <tr>
        <td>SWIN-small</td>
        <td>93.90</td>
        <td>94.9</td>
        <td>71.00</td>
        <td>95.00</td>
        <td>93.00</td>
        <td>94.10</td>
    </tr>
    <tr>
        <td>SWIN-base</td>
        <td>94.00</td>
        <td>96.70</td>
        <td>72.00</td>
        <td>95.00</td>
        <td>94.00</td>
        <td>94.90</td>
    </tr>
    <tr>
        <td>CvT-small</td>
        <td>92.80</td>
        <td>95.70</td>
        <td>73.00</td>
        <td>96.00</td>
        <td>96.00</td>
        <td>95.50</td>
    </tr>
    <tr>
        <td>SGADA \cite{sgada2021}</td>
        <td>80.10</td>
        <td>94.24</td>
        <td>87.13</td>
        <td>94.44</td>
        <td>92.03</td>
        <td>91.20</td>
    </tr>
</table>

#### Table 2 [Accuracy values with and without the pseudolabelling technique]
<table>
    <tr>
        <td>Method Name</td>
        <td>Without 2-way center aware labelling</td>
        <td>With 2-way center aware labelling</td>
    </tr>
    <tr>
        <td>DeiT-small</td>
        <td>94.4</td>
        <td>95.9</td>
    </tr>
    <tr>
        <td>DeiT-base</td>
        <td>96.1</td>
        <td>96.4</td>
    </tr>
    <tr>
        <td>CvT</td>
        <td>93.6</td>
        <td>95.5</td>
    </tr>
</table>

#### Table 3 [Accuracy values with and without the discriminator added to the network for DeiT-small.]
<table>
    <tr>
        <td>Method Name</td>
        <td>Accuracy</td>
    </tr>
    <tr>
        <td>CDTrans</td>
        <td>95.9</td>
    </tr>
    <tr>
        <td>CDTrans + discriminator</td>
        <td>94.1</td>
    </tr>
</table>

Trained model models can be found here [ModelZoo](https://mbzuaiac.sharepoint.com/:f:/s/AI702_group7/Eh2v0jwRrltBh6D-XMNL0yMBLFnLLtxdrAfQ_yqLLiTp4Q?e=HbXFDA)

## Requirements
### Installation
```bash
pip install -r requirements.txt
(Python version is the 3.7 and the GPU is the Quadro RTX 6000 with cuda 12.0, cudatoolkit 12.0)
```
### Prepare Datasets
Before running the training code, make sure that DATASETDIR environment variable is set to your dataset directory.
```
$ export DATASETDIR="/path/to/dataset/dir"
```

Download FLIR ADAS dataset: [Link](https://www.flir.eu/oem/adas/adas-dataset-form/)
- Download MS-COCO dataset: 
  - [Train images](http://images.cocodataset.org/zips/train2017.zip) 
  - [Val images](http://images.cocodataset.org/zips/val2017.zip) 
  - [Train/Val annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- After you downloaded the datasets, extract them to `DATASETDIR`.
- Crop annotated objects (for bicycle, car and person classes only) using the command below:
```bash
$ python utils/prepare_dataset.py
```
After the preparation steps, your dataset folder should be in the following structure.
```
DATASETDIR
└── sgada_data
    ├── flir
    │   ├── train
    │   │   ├── bicycle
    │   │   ├── car
    │   │   └── person
    │   ├── val
    │   │   ├── bicycle
    │   │   ├── car
    │   │   └── person
    ├── mscoco
    |    ├── train
    |    │   ├── bicycle
    |    │   ├── car
    |    │   └── person
    |    └── val
    |        ├── bicycle
    |        ├── car
    |        └── person
	├── flir_train.txt
	├── flir_val.txt
	├── mscoco_train.txt
	└── mscoco_val.txt
```
### Prepare DeiT-trained Models
For fair comparison in the pre-training data set, we use the DeiT parameter init our model based on ViT. 
You need to download the ImageNet pretrained transformer model : [DeiT-Small](https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth), [DeiT-Base](https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth), [cvt](https://1drv.ms/u/s!AhIXJn_J-blW9RzF3rMW7SsLHa8h?e=blQ0Al), [swin_small](https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth), [swin_base](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth) and move them to the `./data/pretrainModel` directory.

## Training
We utilize 1 GPU for both pre-training and UDA stages.
Before running the scripts make sure to update the data directories in the scripts/[pretrain/uda].sh files as follows
```
DATASETS.ROOT_TRAIN_DIR '/path/to/mscoco_train.txt' 
DATASETS.ROOT_TRAIN_DIR2 '/path/to/flir_train.txt' 
DATASETS.ROOT_TEST_DIR '/path/to/flir_val.txt'
```

# Scripts.
Command input paradigm

`bash scripts/run_[pretrain/uda].sh [deit_base/deit_small/cvt/swin_small/swin_base] run`

## For example
To run DeiT-Base
```bash

bash scripts/run_pretrain.sh deit_base run_1
bash scripts/run_uda.sh deit_base run_1


## Evaluation
```bash

python test.py --config_file 'configs/uda.yml' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('/path/to/trained/weight.pth')" DATASETS.NAMES 'cocoflir' DATASETS.NAMES2 'cocoflir' OUTPUT_DIR '../logs/uda/deit_base/test/' DATASETS.ROOT_TRAIN_DIR '/path/to/mscoco_train.txt' DATASETS.ROOT_TRAIN_DIR2 '/path/to/flir_train.txt' DATASETS.ROOT_TEST_DIR '/path/to/flir_val.txt'  
```
The output logs will be saved at ../logs/[pretrain/uda]/[deit_base/deit_small/cvt/swin_small/swin_base]/run

## Acknowledgement

Codebase from [CDTrans](https://github.com/CDTrans/CDTrans.git)


