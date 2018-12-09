from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
from PIL import Image
import os
import pickle
from pdb import set_trace as st

class KITTIdataset(Dataset):
    """KITTIdataset"""
    def __init__(self, list_file='train.txt', data_root_path='/mnt/lustre/zhangziqi/Dataset/kitti_eigen_split_3',
                img_size=[128, 416], bundle_size=3, min_kpts_num=150):
        self.data_root_path = data_root_path
        self.img_size = img_size
        self.bundle_size = bundle_size
        self.min_kpts_num = min_kpts_num
        self.frame_pathes = []
        list_file = os.path.join(data_root_path, list_file)
        with open(list_file) as file:
            for line in file:
                frame_path = line.strip()
                seq_path, frame_name = frame_path.split(" ")
                frame_path = os.path.join(seq_path, frame_name)
                self.frame_pathes.append(frame_path)
                # print(frame_path)
        # self.frame_pathes = self.frame_pathes[0:40000:800]

    def __len__(self):
        return len(self.frame_pathes)

    def __getitem__(self, item):
        # read camera intrinsics
        cam_file = os.path.join(self.data_root_path, self.frame_pathes[item]+'_cam.txt')
        with open(cam_file) as file:
            cam_intrinsics = [float(x) for x in next(file).split(',')]
        # camparams = dict(fx=cam_intrinsics[0], cx=cam_intrinsics[2],
        #             fy=cam_intrinsics[4], cy=cam_intrinsics[5])
        camparams = np.asarray(cam_intrinsics).astype(np.float32)

        # read image bundle
        img_file = os.path.join(self.data_root_path, self.frame_pathes[item]+'.jpg')
        frames_cat = np.array(Image.open(img_file))
        # slice the image into #bundle_size number of images
        frame_list = []
        for i in range(self.bundle_size):
            frame_list.append(frames_cat[:,i*self.img_size[1]:(i+1)*self.img_size[1],:])
        frames = np.asarray(frame_list).astype(float).transpose(0, 3, 1, 2)

        kpts_path = os.path.join(self.data_root_path, self.frame_pathes[item]+'.pickle')
        with open(kpts_path, 'rb') as f:
            kpts_dict = pickle.load(f)
        kpts = []
        ref_idx = int((self.bundle_size-1)/2)
        src_idxs = tuple(range(0,ref_idx)) + tuple(range(ref_idx+1,self.bundle_size))
        for src_idx in src_idxs:
            single_pair = kpts_dict[src_idx]
            # [src, ref]
            single_pair = np.array([single_pair[src_idx], single_pair[ref_idx]]).astype(np.float32)
            if single_pair.shape[1]<self.min_kpts_num:
                repeat_time = int(self.min_kpts_num/single_pair.shape[1])+1
                single_pair = np.tile(single_pair, (1,repeat_time,1))
            kpts.append(single_pair[:,:self.min_kpts_num,:])
        kpts = np.array(kpts)
        
        return frames, camparams, kpts

if __name__ == "__main__":
    dataset = KITTIdataset()
    dataset.__getitem__(0)
    for i, data in enumerate(dataset):
        print(data[1])
