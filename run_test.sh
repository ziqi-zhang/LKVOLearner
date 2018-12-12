DATAROOT=/mnt/lustre/zhangziqi/Dataset/kitti_raw_eigen/
PWD=$(pwd)
CKPT=$PWD/checkpoints/check_code/9_model.pth
# CKPT=$PWD/posenet+ddvo.pth
OUTPUT=$PWD/checkpoints/check_code/output.npy
CUDA_VISIBLE_DEVICES=0 nice -10 python src/testKITTI.py --dataset_root $DATAROOT --ckpt_file $CKPT --output_path $OUTPUT --test_file_list test_files_eigen.txt

python2 src/util/eval_depth.py --kitti_dir=$DATAROOT --pred_file=$OUTPUT --test_file_list test_files_eigen.txt
