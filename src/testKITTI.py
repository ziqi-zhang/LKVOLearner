from networks import VggDepthEstimator
from LKVOLearner import FlipLR
import torch
from torch.autograd import Variable
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pdb import set_trace
import cv2

from util.depth_visualize import *

def colored_depthmap(depth, d_min=None, d_max=None):
    import matplotlib.pyplot as plt
    cmap = plt.cm.viridis

    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return (255 * cmap(depth_relative)[:,:,:3]).astype(np.uint8) # H, W, C

def filepath_to_filename(filepath):
    items = filepath.split('/')
    filename = items[0]
    for item in items[1:]:
        filename = filename+'.'+item
    return filename

def read_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    lines = [l.rstrip() for l in lines]
    return lines

def main():

    """
    CUDA_VISIBLE_DEVICES=1 nice -10 python3 testKITTI.py --dataset_root /newfoundland/chaoyang/kitti --ckpt_file /home/chaoyang/LKVOLearner/checkpoints/checkpoints_19_416_scratch/9_model.pth --test_file_list
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", type=str, default="/newfoundland/chaoyang/kitti", help="dataset root path")
    parser.add_argument("--test_file_list", type=str, default="/newfoundland/chaoyang/SfMLearner/data/kitti/test_files_eigen.txt", help="test file list")
    parser.add_argument("--ckpt_file", type=str, default=None, help="checkpoint file")
    parser.add_argument("--output_path", type=str, default="pred_depths", help="output path")
    parser.add_argument("--vis_dir", type=str, default=None)
    parser.add_argument("--use_pp", default=False, action="store_true", help='use post processing')

    FLAGS = parser.parse_args()

    # dataset_root = "/newfoundland/chaoyang/kitti"
    # model_path = "/home/chaoyang/LKVOLearner/checkpoints_new/12_model.pth"
    # test_file_list = "/newfoundland/chaoyang/SfMLearner/data/kitti/test_files_eigen.txt"
    dataset_root = FLAGS.dataset_root
    model_path = FLAGS.ckpt_file
    test_file_list = FLAGS.test_file_list
    output_path = FLAGS.output_path

    img_size = [128, 416]
    vgg_depth_net = VggDepthEstimator(img_size)
    vgg_depth_net.load_state_dict(torch.load(model_path))
    vgg_depth_net.cuda()

    pred_depths, _ = predKITTI(vgg_depth_net, dataset_root, test_file_list, img_size,
                vis_dir=FLAGS.vis_dir, use_pp=FLAGS.use_pp)
    print(pred_depths.shape)
    np.save(output_path, pred_depths)
    # import scipy.io as sio
    # sio.savemat(output_path, {'D': pred_depths})
        # print(pred_depth_pyramid[0].size())
        # plt.imshow(pred_depth_pyramid[0].data.cpu().squeeze().numpy())
        # plt.show()

def vis_heat_map(conv_map, upconv_map, img_size):
    conv_map_vis = []
    for i, map_rgb in enumerate(conv_map):
        map_rgb = tensor2numpy(map_rgb)
        if map_rgb.ndim==3:
            map_rgb = map_rgb[1]
        map_rgb = map_rgb.squeeze()
        map_rgb = cv2.resize(map_rgb, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
        if i<2:
            map_rgb = feat_map2gray(map_rgb)
        else:
            map_rgb = feat_map2gray(map_rgb)
        # use_pp, map_rgb[1] is the original image
        # map_rgb = np.resize(map_rgb, (img_size[0], img_size[1], 3))

        conv_map_vis.append(map_rgb)

    upconv_map_vis = []
    for i, map_rgb in enumerate(upconv_map):
        map_rgb = tensor2numpy(map_rgb)
        if map_rgb.ndim==3:
            map_rgb = map_rgb[1]
        map_rgb = map_rgb.squeeze()
        map_rgb = cv2.resize(map_rgb, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
        if i>4:
            map_rgb = feat_map2gray(map_rgb)
        else:
            map_rgb = feat_map2gray(map_rgb)
        # map_rgb = np.resize(map_rgb, (img_size[0], img_size[1], 3))

        upconv_map_vis.append(map_rgb)

    return conv_map_vis, upconv_map_vis


def predKITTI(model, dataset_root, test_file_list, img_size=[128, 416],
                vis_dir=None, use_pp=True):
    print("Begin test KITTI")

    fliplr = FlipLR(imW=img_size[1], dim_w=2).cuda()

    test_files = read_text_lines(test_file_list)
    pred_disp = []
    raw_images = []
    conv_map_list = []
    upconv_map_list = []
    i = 0
    for filename in test_files:
        filename = filename.split()[0]
        im_path = os.path.join(dataset_root, filename)
        img_pil = Image.open(im_path).resize((img_size[1], img_size[0]), Image.ANTIALIAS)
        # img_pil.save('kitti_test_images/%04d.png'%(i))
        img = np.array(img_pil)
        raw_images.append(img)
        # print(img.shape)
        img = img.transpose(2, 0, 1)
        # print(img.shape)
        # print(filename)
        img_var = Variable(torch.from_numpy(img).float().cuda(), volatile=True)


        if use_pp:
            # flip image
            img_vars = (torch.cat((fliplr(img_var).unsqueeze(0), img_var.unsqueeze(0)), 0)-127)/127
            pred_disp_pyramid, conv_map, upconv_map = model.forward(img_vars)
            disp = pred_disp_pyramid[0]
            # print(depth.size())
            disp_mean = (fliplr(disp[0:1, :, :]) + disp[1:2, :, :])*.5
            disp_mean = disp_mean.data.cpu().squeeze().numpy()
            pred_disp.append(disp_mean)
            # compute mean
        else:
            pred_disp_pyramid, conv_map, upconv_map = model.forward((img_var.unsqueeze(0)-127)/127)
            pred_disp.append(pred_disp_pyramid[0].data.cpu().squeeze().numpy())
        if vis_dir is not None:
            disp_vis = colored_depthmap(pred_disp[-1])
            vis_img = np.hstack([img_pil, disp_vis])
            vis_file_name = filepath_to_filename(filename)
            vis_file_path = os.path.join(vis_dir, vis_file_name)
            Image.fromarray(vis_img).save(vis_file_path)

        # set_trace()
        conv_map_vis, upconv_map_vis = vis_heat_map(conv_map, upconv_map, img_size)
        conv_map_vis = np.vstack(conv_map_vis)
        upconv_map_vis = np.vstack(upconv_map_vis)
        conv_map_list.append(np.asarray(conv_map_vis))
        upconv_map_list.append(np.asarray(upconv_map_vis))
        i = i+1
        if i>3:
            break

    pred_disp = np.asarray(pred_disp)
    raw_images = np.asarray(raw_images)
    return 1/pred_disp, raw_images, conv_map_list, upconv_map_list

if __name__=='__main__':
    main()
