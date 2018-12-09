srun -p Segmentation \
python -u src/ComputeKpts.py \
--data-root-path /mnt/lustre/zhangziqi/Dataset/kitti_eigen_split_3 \
--img-height 128 \
--img-width 416 \
--min-kpts-num 100 \
--seq-length 3 \
--num-threads 1
