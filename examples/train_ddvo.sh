
MODEL_ID=9
PWD=$(pwd)
mkdir $PWD/checkpoints/
EXPNAME=scratch_ddvo
CHECKPOINT_DIR=$PWD/checkpoints/$EXPNAME
mkdir $CHECKPOINT_DIR

DATAROOT_DIR=/mnt/lustre/zhangziqi/Dataset/kitti_eigen_split_3

srun -p Segmentation -n1 --mpi=pmi2 --gres=gpu:1 \
--ntasks-per-node=1 --job-name=ddvo \
python3 -u src/train_main_ddvo.py --dataroot $DATAROOT_DIR\
 --checkpoints_dir $CHECKPOINT_DIR --which_epoch -1 --save_latest_freq 1000\
  --batchSize 1 --display_freq 50 --name $EXPNAME\
  --lk_level 5 --lambda_S 0.01 --smooth_term 2nd --use_ssim --display_port 8009 --epoch_num 10
  --lr 0.00001
