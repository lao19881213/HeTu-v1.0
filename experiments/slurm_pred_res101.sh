#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8g
#SBATCH --partition=inspur-gpu-opa
#SBATCH --export=ALL

source /home/blao/rgz_resnet_fpn/bashrc
#module load tensorflow/1.12.0-py36-gpu
#module load nccl
#module load openmpi

#export PYTHONPATH=$PYTHONPATH:/flush1/wu082/proj/rgz_resnet_fpn/pyenv
#source /flush1/wu082/venvs/rgz_resnet_fpn_env/bin/activate
LOG_DIR=/home/blao/rgz_resnet_fpn/train_log/$SLURM_JOB_ID
cd /home/blao/rgz_resnet_fpn

#source /flush1/wu082/venvs/rgz_resnet_fpn_env/bin/activate
#INPUT_IMAGE=/home/blao/rgz_resnet_fpn/data/testD1new/FIRSTJ162040.4+293911_logminmax_new.png
#INPUT_IMAGE=/home/blao/tf-faster-rcnn/data/demo/FIRSTJ081700.6+571626_logminmax_new.png 
INPUT_IMAGE=/home/blao/rgz_resnet_fpn/data/testD1new/FIRSTJ230229.6+054819_logminmax_new.png #/o9000/MWA/GLEAM/G0008_images/deep_learn/data/GLEAM20130810_1060208376_200-231MHz_1680_1456.png #/home/blao/rgz_resnet_fpn/data/testD1new/FIRSTJ163826.9+343128_logminmax_new.png
python train.py --predict $INPUT_IMAGE  \
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
