model=$1
if [ ! -n "$1" ]
then 
    echo 'pelease input the model para: {deit_base, deit_small, swin_base, swin_small}'
    exit 8
fi
if [ $model == 'deit_base' ]
then
    model_type='vit_base_patch16_224_TransReID'
    pretrain_model='deit_base_distilled_patch16_224-df68dfff.pth'
elif [ $model == 'swin_small' ]
then
    model='swin_small'
    model_type='swin_small_patch4_window7_224_TransReID'
    pretrain_model='swin_small_patch4_window7_224_22k.pth'
elif [ $model == 'deit_small' ]
then
    model='deit_small'
    model_type='uda_vit_small_patch16_224_TransReID'
    pretrain_model='uda_deit_small_distilled_patch16_224-649709d9.pth'
else
    model='swin_base'
    model_type='swin_base_patch4_window7_224_TransReID'
    pretrain_model='swin_base_patch4_window7_224_22k.pth'
fi
python test.py --config_file configs/uda.yml MODEL.DEVICE_ID "('0')" \
OUTPUT_DIR '../logs/target/perclass/'$model'/coco-flir/flir' \
DATASETS.ROOT_TRAIN_DIR '../test/sgada_data/mscoco.txt' \
DATASETS.ROOT_TRAIN_DIR2 '../test/sgada_data/flir.txt' \
DATASETS.ROOT_TEST_DIR '../test/sgada_data/flir.txt'   \
MODEL.Transformer_TYPE $model_type \
DATASETS.NAMES "cocoflir" DATASETS.NAMES2 "cocoflir" \
TEST.WEIGHT '../logs/uda/'$model'/coco-flir/mscoco/transformer_best_model.pth' \


