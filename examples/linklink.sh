
BS=8
GPU=8
TOTAL_BS=`echo "$GPU*$BS"|bc`

PWD=$(pwd)
mkdir -p $PWD/checkpoints/
# EXPNAME=linklink_$TOTAL_BS
EXPNAME=norm_baseline
CHECKPOINT_DIR=$PWD/checkpoints/$EXPNAME
mkdir -p $CHECKPOINT_DIR

POSENET_CKPT_DIR=$PWD/checkpoints/posenet
PRETRAIN_MODEL_ID=9
cp $(printf "%s/%s_pose_net.pth" "$POSENET_CKPT_DIR" "$PRETRAIN_MODEL_ID") $CHECKPOINT_DIR/pose_net.pth
cp $(printf "%s/%s_depth_net.pth" "$POSENET_CKPT_DIR" "$PRETRAIN_MODEL_ID") $CHECKPOINT_DIR/depth_net.pth

DATAROOT_DIR=/mnt/lustre/zhangziqi/Dataset/kitti_eigen_split_3/

srun -p Segmentation -n$GPU --mpi=pmi2 --gres=gpu:$GPU \
--ntasks-per-node=$GPU --job-name=ddvo \
python -u src/train_main_ddvo.py --dataroot $DATAROOT_DIR \
--checkpoints_dir $CHECKPOINT_DIR --which_epoch -1 --save_latest_freq 1000 \
--batchSize $BS --display_freq 100 --name $EXPNAME \
--lk_level 5 --lambda_S 0.01 --smooth_term 2nd \
--display_port 8009 --epoch_num 10 \
--use_ssim --lr `echo "0.00001*$TOTAL_BS"|bc` \
--val_data_root_path /mnt/lustre/zhangziqi/Dataset/kitti_raw_eigen \
