#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8g
#SBATCH --partition=inspur-gpu-ib
#SBATCH --export=ALL

source /home/blao/rgz_resnet_fpn/bashrc
#module load tensorflow/1.12.0-py36-gpu
#module load nccl
#module load openmpi

#export PYTHONPATH=$PYTHONPATH:/flush1/wu082/proj/rgz_resnet_fpn/pyenv
#source /flush1/wu082/venvs/rgz_resnet_fpn_env/bin/activate
LOG_DIR=/home/blao/rgz_resnet_fpn/train_log/$SLURM_JOB_ID
cd /home/blao/rgz_resnet_fpn
start_time=`date +%s`
time python train.py --logdir $LOG_DIR --config \
        MODE_MASK=False MODE_FPN=True \
        DATA.BASEDIR=./data \
        BACKBONE.WEIGHTS=./weights/pretrained/ImageNet-R101-AlignPadding.npz \
        DATA.TRAIN=trainD1new DATA.VAL=testD1new \
        PREPROC.TRAIN_SHORT_EDGE_SIZE=600,600 \
        PREPROC.TEST_SHORT_EDGE_SIZE=600 \
	TRAIN.LR_SCHEDULE=20000,30000,40000 \
        DATA.CLASS_NAMES='cs','fr1','fr2','core_jet' \
        BACKBONE.RESNET_NUM_BLOCKS=3,4,23,3 \
        TRAIN.NUM_GPUS=1

end_time=`date +%s`
duration=`echo "$end_time-$start_time" | bc -l`
echo "Total runtime = $duration sec"
