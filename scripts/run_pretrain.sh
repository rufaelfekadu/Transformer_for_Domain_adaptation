model=$1
run=$2
if [ ! -n "$1" ]
then 
    echo 'pelease input the model para: {deit_base, deit_small,swin_small,swin_base,cvt}'
    exit 8
fi
if [ $model == 'deit_base' ]
then
    model_type='vit_base_patch16_224_TransReID'
    pretrain_model='deit_base_distilled_patch16_224-df68dfff.pth'
    in_planes=768
    gpus="('0,1')"
elif [ $model == 'swin_small' ]
then
    model_type='swin_small_patch4_window7_224_TransReID'
    pretrain_model='swin_small_patch4_window7_224_22k.pth'
    in_planes=768
    gpus="('0')"
elif [ $model == 'swin_base' ]
then
    model_type='swin_base_patch4_window7_224_TransReID'
    pretrain_model='swin_base_patch4_window7_224_22k.pth'
    in_planes=1024
    gpus="('0,1')"
elif [ $model == 'cvt' ]
then
    model_type='cvt_21_224_TransReID'
    pretrain_model='CvT-21-224x224-IN-1k.pth'
    in_planes=384
    gpus="('0,1')"
else
    model_type='vit_small_patch16_224_TransReID'
    pretrain_model='deit_small_distilled_patch16_224-649709d9.pth'
    in_planes=384
    gpus="('0')"
fi

python test.py --config_file configs/pretrain.yml MODEL.DEVICE_ID $gpus \
OUTPUT_DIR '../logs/pretrain/'$model'/'$run \
DATASETS.ROOT_TRAIN_DIR '../dataset/sgada_data/flir_train.txt' \
DATASETS.ROOT_TRAIN_DIR2 '../dataset/sgada_data/flir_train.txt' \
DATASETS.ROOT_TEST_DIR '../dataset/sgada_data/flir_val.txt' \
DATASETS.NAMES "cocoflir" DATASETS.NAMES2 "cocoflir" \
MODEL.Transformer_TYPE $model_type \
MODEL.PRETRAIN_PATH './data/pretrainModel/'$pretrain_model \
TEST.WEIGHT '../logs/pretrain/'$model'/1/transformer_best_model.pth' \
MODEL.IN_PLANES $in_planes \