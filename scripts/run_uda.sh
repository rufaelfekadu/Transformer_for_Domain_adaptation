model=$1
run=$2
if [ ! -n "$1" ]
then 
    echo 'pelease input the model para: {deit_base, deit_small, swin_small, swin_base, cvt}'
    exit 8
fi
if [ $model == 'deit_base' ]
then
    model_type='uda_vit_base_patch16_224_TransReID'
    gpus="('0,1')"
    in_planes=768
elif [ $model == 'swin_small' ]
then
    model='swin_small'
    model_type='uda_swin_small_patch4_window7_224_TransReID'
    gpus="('0')"
    in_planes=768
elif [ $model == 'cvt' ]
then
    model='cvt'
    model_type='uda_cvt_21_224_TransReID'
    gpus="('0')"
    in_planes=384
elif [ $model == 'swin_base' ]
then
    model='swin_base'
    model_type='uda_swin_base_patch4_window7_224_TransReID'
    in_planes=1024
    gpus="('0,1')"
else
    model='deit_small'
    model_type='uda_vit_small_patch16_224_TransReID'
    gpus="('0')"
    in_planes=384
fi

python test.py --config_file configs/uda.yml MODEL.DEVICE_ID $gpus \
OUTPUT_DIR '../logs/uda/'$model'/'$run \
DATASETS.ROOT_TRAIN_DIR '../dataset/sgada_data/mscoco_train.txt' \
DATASETS.ROOT_TRAIN_DIR2 '../dataset/sgada_data/flir_train.txt' \
DATASETS.ROOT_TEST_DIR '../dataset/sgada_data/flir_val.txt' \
DATASETS.NAMES "cocoflir" DATASETS.NAMES2 "cocoflir" \
MODEL.Transformer_TYPE $model_type \
MODEL.IN_PLANES $in_planes \
MODEL.PRETRAIN_PATH '../logs/pretrain/'$model'/1/transformer_best_model.pth' \
# MODEL.USE_DISC 'True' \
# MODEL.PRETRAIN_PATH "./data/pretrainModel/swin_base_patch4_window7_224_22k.pth" \

# DATASETS.ROOT_TRAIN_DIR '../dataset/sgada_data/mscoco_train.txt' \
# DATASETS.ROOT_TRAIN_DIR2 '../dataset/sgada_data/flir_train.txt' \
# DATASETS.ROOT_TEST_DIR '../dataset/sgada_data/flir_val.txt' \
# DATASETS.ROOT_TRAIN_DIR './data/cocoflir_1000/mscoco.txt' \
# DATASETS.ROOT_TRAIN_DIR2 './data/cocoflir_1000/flir.txt' \
# DATASETS.ROOT_TEST_DIR './data/cocoflir_1000/flir.txt' \