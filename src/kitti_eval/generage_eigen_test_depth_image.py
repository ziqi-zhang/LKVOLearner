import os
from PIL import Image
import numpy as np
from pdb import set_trace
from depth_evaluation_utils import *
import matplotlib.pyplot as plt

cmap = plt.cm.viridis

kitti_raw_dir = '/mnt/lustre/zhangziqi/Dataset/kitti_raw_data'
eigen_test_file = '/mnt/lustre/zhangziqi/Dataset/kitti_raw_eigen/list/eigen_test_files.txt'
save_dir = '/mnt/lustre/zhangziqi/Dataset/kitti_eigen_split_test'

def read_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    lines = [l.rstrip() for l in lines]
    return lines


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)
    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    return depth

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return (255 * cmap(depth_relative)[:,:,:3]).astype(np.uint8) # H, W, C


def depth_save(depth, filename):
    depth = depth*256
    depth_png = depth.astype(np.uint32)
    assert np.max(depth_png)>255
    Image.fromarray(depth_png).save(filename)

def main():
    lines = read_text_lines(eigen_test_file)
    cnt = 0
    npy_result = []

    for i, line in enumerate(lines):
        # if i>2:
        #     break
        file_path = line.split()[0]
        date, seq, cam_id, _, file = file_path.split('/')
        file = file.split('.')[0]
        image_path = os.path.join(kitti_raw_dir, file_path)
        calib_dir = os.path.join(kitti_raw_dir, date)

        velo_file_name = os.path.join(kitti_raw_dir, date, seq, 'velodyne_points/data', '%s.bin'%file)
        assert os.path.exists(image_path) and os.path.exists(velo_file_name) and \
            os.path.exists(calib_dir)
        image = np.array(Image.open(image_path))
        depth = generate_depth_map(calib_dir, velo_file_name, image.shape[:2])
        depth_vis = colored_depthmap(depth)

        save_name = "%s.%s.%s.%s.%s"%(date, seq, cam_id, 'data', file)
        depth_name = save_name+'.png'

        depth_vis_name = save_name+'.png'
        image_save_path = os.path.join(save_dir, 'image', depth_name)
        vis_save_path = os.path.join(save_dir, 'visualize', depth_vis_name)
        # npy_save_path = os.path.join(save_dir, 'npy', npy_name)
        depth_save(depth, image_save_path)
        Image.fromarray(depth_vis).save(vis_save_path)
        npy_result.append(depth)
    npy_name = 'depth_gt.npy'
    npy_result = np.array(npy_result)
    np.save(npy_name, npy_result)


if __name__=='__main__':
    main()
