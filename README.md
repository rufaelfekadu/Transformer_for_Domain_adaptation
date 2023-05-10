# Transformer for Unsupervised Domain Adaptation

## Introduction
This repo containes the implementation of a crossdomain transformer architecture adopted from [CDTrans](https://github.com/CDTrans/CDTrans.git) for domain adaptation in Urban Road Scene Perception
```latex
\input{plots_and_tables/tables.tex}
```
## Results
#### Table 1 [UDA results on Office-31]


#### Table 2 [UDA results on Office-Home]


#### Table 3 [UDA results on VisDA-2017]


#### Table 4 [UDA results on DomainNet]

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
    │   ├── test_wconf_wdomain_weights.txt
    │   └── validation_wconf_wdomain_weights.txt
    └── mscoco
        ├── train
        │   ├── bicycle
        │   ├── car
        │   └── person
        └── val
            ├── bicycle
            ├── car
            └── person
```
### Prepare DeiT-trained Models
For fair comparison in the pre-training data set, we use the DeiT parameter init our model based on ViT. 
You need to download the ImageNet pretrained transformer model : [DeiT-Small](https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth), [DeiT-Base](https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth) and move them to the `./data/pretrainModel` directory.

## Training
We utilize 1 GPU for pre-training and 2 GPUs for UDA, each with 16G of memory.

# Scripts.
Command input paradigm

`bash scripts/[pretrain/uda]/[office31/officehome/visda/domainnet]/run_*.sh [deit_base/deit_small]`
## For example
DeiT-Base scripts
```bash

# Office-31     Source: Amazon   ->  Target: Dslr, Webcam
bash scripts/pretrain/office31/run_office_amazon.sh deit_base
bash scripts/uda/office31/run_office_amazon.sh deit_base

#Office-Home    Source: Art      ->  Target: Clipart, Product, Real_World
bash scripts/pretrain/officehome/run_officehome_Ar.sh deit_base
bash scripts/uda/officehome/run_officehome_Ar.sh deit_base

# VisDA-2017    Source: train    ->  Target: validation
bash scripts/pretrain/visda/run_visda.sh deit_base
bash scripts/uda/visda/run_visda.sh deit_base

# DomainNet     Source: Clipart  ->  Target: painting, quickdraw, real, sketch, infograph
bash scripts/pretrain/domainnet/run_domainnet_clp.sh deit_base
bash scripts/uda/domainnet/run_domainnet_clp.sh deit_base
```
DeiT-Small scripts
Replace deit_base with deit_small to run DeiT-Small results. An example of training on office-31 is as follows:
```bash
# Office-31     Source: Amazon   ->  Target: Dslr, Webcam
bash scripts/pretrain/office31/run_office_amazon.sh deit_small
bash scripts/uda/office31/run_office_amazon.sh deit_small
```

## Evaluation
```bash
# For example VisDA-2017
python test.py --config_file 'configs/uda.yml' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/uda/vit_base/visda/transformer_best_model.pth')" DATASETS.NAMES 'VisDA' DATASETS.NAMES2 'VisDA' OUTPUT_DIR '../logs/uda/vit_base/visda/' DATASETS.ROOT_TRAIN_DIR './data/visda/train/train_image_list.txt' DATASETS.ROOT_TRAIN_DIR2 './data/visda/train/train_image_list.txt' DATASETS.ROOT_TEST_DIR './data/visda/validation/valid_image_list.txt'  
```

## Acknowledgement

Codebase from [TransReID](https://github.com/damo-cv/TransReID)


