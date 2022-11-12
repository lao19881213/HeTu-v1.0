#!/bin/bash

#SBATCH --nodes=1
##SBATCH --time=16:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8g
#SBATCH --partition=inspur-gpu-ib
#SBATCH --export=ALL

source /home/blao/rgz_resnet_fpn/bashrc

cd /home/blao/rgz_resnet_fpn

INPUT_IMAGE_DIR=/home/blao/rgz_resnet_fpn/data/mwa_big_fov_test
start_time=`date +%s`
python train.py --batchpred $INPUT_IMAGE_DIR  \
        --sourcefounding 1 \
        --load /home/blao/rgz_resnet_fpn/train_log/10479/model-40000 \
        --config MODE_MASK=False MODE_FPN=True \
        DATA.BASEDIR=./data \
        BACKBONE.WEIGHTS=./weights/pretrained/ImageNet-R101-AlignPadding.npz \
        DATA.TRAIN=trainD1new DATA.VAL=testD1new \
        PREPROC.TRAIN_SHORT_EDGE_SIZE=600,600 \
        PREPROC.TEST_SHORT_EDGE_SIZE=600 \
	TRAIN.LR_SCHEDULE=20000,30000,40000 \
	TEST.RESULT_SCORE_THRESH_VIS=0.7 \
	TEST.RESULT_SCORE_THRESH=0.7 \
        DATA.NUM_CATEGORY=4 BACKBONE.RESNET_NUM_BLOCKS=3,4,23,3 \
        DATA.CLASS_NAMES='cs','fr1','fr2','core_jet'  

end_time=`date +%s`
duration=`echo "$end_time-$start_time" | bc -l`
echo "Total runtime = $duration sec"
