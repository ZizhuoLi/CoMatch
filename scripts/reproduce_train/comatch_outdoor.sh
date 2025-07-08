#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"


export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

# to reproduced the results in our paper, please use:

n_nodes=1
n_gpus_per_node=4 
torch_num_workers=4 
batch_size=3 
pin_memory=true


DATASET=outdoor

TRAIN_IMG_SIZE=832                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     #832 800 768

DEFAULT_EXP_NAME=''
EXP_NAME=${1:-${DEFAULT_EXP_NAME}}
exp_name="outdoor-ds-${TRAIN_IMG_SIZE}-bs=$(($n_gpus_per_node * $n_nodes * $batch_size))-${EXP_NAME}"
if [ $DATASET = "indoor" ]; then
    main_cfg_path="configs/loftr/outdoor/loftr_ds_dense.py"
    data_cfg_path="configs/data/megadepth_trainval_${TRAIN_IMG_SIZE}.py"
elif [ $DATASET = "outdoor" ]; then
    main_cfg_path="configs/loftr/comatch_full.py"
    data_cfg_path="configs/data/megadepth_trainval_${TRAIN_IMG_SIZE}.py"
fi

python -u ./train.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=100 \
    --flush_logs_every_n_steps=1000 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=10 \
    --benchmark=True \
    --max_epochs=30 \
    --thr 0.1 \
    --train_coarse_percent 0.3 \
    --disable_mp \
