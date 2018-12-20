import torch
from torch import optim
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
import scipy.io as sio
import os
import csv
import cv2
import shutil

from LKVOLearner import LKVOLearner
from KITTIdataset import KITTIdataset
from testKITTI import predKITTI
from util.eval_depth import evaluate
from util.depth_visualize import vis_depthmap

from collections import OrderedDict
from options.train_options import TrainOptions
from util.visualizer import Visualizer

from timeit import default_timer as timer
# data_root_path = '/newfoundland/chaoyang/SfMLearner/data_kitti'

from pdb import set_trace as st
from util.logger import Logger
from util.util import *
from PIL import Image
from multiprocessing import Pool
fieldnames = ['abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all',
                'a1', 'a2', 'a3']

import linklink as link
from distributed_utils import dist_init, reduce_gradients, DistModule


def save_val_images(data):
        i, (depth, raw_img, error, val_vis_dir) = data
        error = error.transpose(1,0)
        depth = cv2.resize(depth, error.shape)
        raw_img = cv2.resize(raw_img, error.shape)
        error = error.transpose(1,0)
        depth = vis_depthmap(1/depth)*255
        # mask = error!=0
        # error[mask] = 1/error[mask]
        mask = (error == 0)
        error = vis_depthmap(error)*255
        error[mask] = 0
        img_path = os.path.join(val_vis_dir, "%03d.png"%i)
        img = np.vstack((raw_img, depth, error))
        img = img.astype(np.uint8)
        save_image(img, img_path)

def validate(lkvolearner, dataset_root, epoch, vis_dir=None,
                img_size=[128, 416]):

    vgg_depth_net = lkvolearner.lkvo.module.depth_net
    test_file_list = os.path.join(dataset_root, 'list', 'eigen_test_files.txt')
    print("Predicting validate set")
    pred_depths, raw_images = predKITTI(vgg_depth_net, dataset_root, test_file_list, img_size,
                use_pp=True)

    # pred_depths = np.zeros((697, 128, 375))+1
    print("Evaluating")
    abs_rel, sq_rel, rms, log_rms, d1_all, a1, a2, a3, error_maps = \
        evaluate(pred_depths, test_file_list, dataset_root)
    assert pred_depths.shape[0] == raw_images.shape[0] == error_maps.shape[0]
    n = pred_depths.shape[0]
    if vis_dir is not None:
        val_vis_dir = os.path.join(vis_dir, "val_%02d"%epoch)
        mkdir(val_vis_dir)

        pred_depths = list(pred_depths)
        raw_images = list(raw_images)
        error_maps = list(error_maps)
        data_pack = list(enumerate(zip(pred_depths, raw_images, error_maps, [val_vis_dir for _ in range(n)])))
        pool = Pool(20)
        pool.map(save_val_images, data_pack)
        pool.close()
        pool.join()

    return abs_rel, sq_rel, rms, log_rms, d1_all, a1, a2, a3

def main():
    rank, world_size = dist_init()

    opt = TrainOptions().parse(rank)
    if rank==0:
        logger = Logger(opt.tf_log_dir)
    img_size = [opt.imH, opt.imW]

    if rank==0:
        test_csv = os.path.join(opt.checkpoints_dir, 'test.csv')
        with open(test_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    # visualizer = Visualizer(opt)

    dataset = KITTIdataset(data_root_path=opt.dataroot, img_size=img_size, bundle_size=3)
    dataloader = DataLoader(dataset, batch_size=opt.batchSize,
                            shuffle=False, num_workers=opt.nThreads, pin_memory=True)



    lkvolearner = LKVOLearner(img_size=img_size, ref_frame_idx=1, lambda_S=opt.lambda_S,
            smooth_term = opt.smooth_term,
            use_ssim=opt.use_ssim)
    lkvolearner.init_weights()


    if opt.which_epoch >= 0:
        print("load pretrained model")
        # lkvolearner.load_model(os.path.join(opt.checkpoints_dir, '%s_model.pth' % (opt.which_epoch)))
        lkvolearner.load_model(os.path.join(opt.checkpoints_dir, '%s_model.pth' % (opt.which_epoch)),
                            os.path.join(opt.checkpoints_dir, 'pose_net.pth'))
    else:
        print("load pretrained models")
        lkvolearner.load_model(os.path.join(opt.checkpoints_dir, 'depth_net.pth'),
                            os.path.join(opt.checkpoints_dir, 'pose_net.pth'))

    # lkvolearner.cuda()
    lkvolearner.dist_model()


    ref_frame_idx = 1

    optimizer = optim.Adam(lkvolearner.get_parameters(), lr=.0001)

    step_num = 0



    for epoch in range(max(0, opt.which_epoch), opt.epoch_num+1):
        if rank==0:
            vis_dir = os.path.join(opt.vis_dir, "train_%02d"%epoch)
            mkdir(vis_dir)
            t = timer()
        for ii, data in enumerate(dataloader):
            optimizer.zero_grad()
            frames = Variable(data[0].float().cuda())
            camparams = Variable(data[1].float().cuda())
            cost, photometric_cost, smoothness_cost, inv_depths, \
            frame_list, inv_depth_list, warp_img_list = \
                lkvolearner.forward(frames, camparams, \
                                    max_lk_iter_num=opt.max_lk_iter_num, \
                                    lk_level=opt.lk_level)
            # print(frames.size())
            # print(inv_depths.size())
            cost_ = cost.data.cpu()
            inv_depths_mean = inv_depths.mean().data.cpu().numpy()
            # if np.isnan(cost_.numpy()) or np.isinf(cost_.numpy()) or inv_depths_mean<1 or inv_depths_mean>7:
            #     # lkvolearner.save_model(os.path.join(opt.checkpoints_dir, '%s_model.pth' % (epoch)))
            #     print("detect nan or inf-----!!!!! %f" %(inv_depths_mean))
            #     continue

            # print(cost)
            # print(inv_depth_pyramid)
            cost.backward()
            reduce_gradients(lkvolearner.lkvo)
            optimizer.step()
            step_num+=1

            if rank==0:
                if np.mod(step_num, opt.print_freq)==0:
                    elapsed_time = timer()-t
                    print('epoch %s[%s/%s], ... elapsed time: %f (s)' % (epoch, step_num, int(len(dataset)/opt.batchSize), elapsed_time))
                    print(inv_depths_mean)
                    t = timer()
                    print("Print: photometric_cost {:.3f}, smoothness_cost {:.3f}, cost {:.3f}".format(photometric_cost.data.cpu().item(),
                            smoothness_cost.data.cpu().item(), cost.data.cpu().item()))
                    # visualizer.plot_current_errors(step_num, 1, opt,
                    #             OrderedDict([('photometric_cost', photometric_cost.data.cpu()[0]),
                    #              ('smoothness_cost', smoothness_cost.data.cpu()[0]),
                    #              ('cost', cost.data.cpu()[0])]))

                    logger.add_scalar('train/photo', photometric_cost.data.cpu().item(), step_num)
                    logger.add_scalar('train/smooth',smoothness_cost.data.cpu().item(), step_num)
                    logger.add_scalar('train/cost', cost.data.cpu().item(), step_num)

                if np.mod(step_num, opt.display_freq)==0:
                    # frame_vis = frames.data[:,1,:,:,:].permute(0,2,3,1).contiguous().view(-1,opt.imW, 3).cpu().numpy().astype(np.uint8)
                    # depth_vis = vis_depthmap(inv_depths.data[:,1,:,:].contiguous().view(-1,opt.imW).cpu()).numpy().astype(np.uint8)
                    frame_vis_list = [frame.data.permute(1,2,0).contiguous().cpu().numpy().astype(np.uint8) for frame in frame_list]
                    depth_vis_list = [(vis_depthmap(depth.data.cpu().numpy())*255).astype(np.uint8) for depth in inv_depth_list]
                    print("Display: photometric_cost {:.3f}, smoothness_cost {:.3f}, cost {:.3f}".format(photometric_cost.data.cpu().item(),
                            smoothness_cost.data.cpu().item(), cost.data.cpu().item()))
                    warp_img_list = [img.cpu().numpy() for img in warp_img_list]
                    warp_img_list = [img*255/(img.max()-img.min()+.00001) for img in warp_img_list]
                    warp_img_list = [img.transpose((0,2,3,1)).astype(np.uint8) for img in warp_img_list]
                    zeros = np.zeros(warp_img_list[0][0].shape)
                    # visualizer.display_current_results(
                    #                 OrderedDict([('%s frame' % (opt.name), frame_vis),
                    #                         ('%s inv_depth' % (opt.name), depth_vis)]),
                    #                         epoch)

                    left_vis = np.vstack([frame_vis_list[0], depth_vis_list[0], warp_img_list[0][0], zeros])
                    mid_vis = np.vstack([frame_vis_list[1], depth_vis_list[1], warp_img_list[1][0], warp_img_list[1][1]])
                    right_vis = np.vstack([frame_vis_list[2], depth_vis_list[2], warp_img_list[2][0], zeros])
                    result_vis = np.hstack([left_vis, mid_vis, right_vis]).astype(np.uint8)
                    save_image(result_vis, os.path.join(vis_dir, 'depth_%05d.png'%step_num))
                    # sio.savemat(os.path.join(opt.checkpoints_dir, 'depth_%s.mat' % (step_num)),
                    #     {'D': inv_depths.data.cpu().numpy(),
                    #      'I': frame_vis})

                if np.mod(step_num, opt.save_latest_freq)==0:
                    print("cache model....")
                    lkvolearner.save_model(os.path.join(opt.checkpoints_dir, '%s_model.pth' % (epoch)))
                    lkvolearner.cuda()
                    print('..... saved')

                if np.mod(step_num, opt.print_freq)==0:
                    print()

        if rank==0:
            #eval
            abs_rel, sq_rel, rms, log_rms, d1_all, a1, a2, a3 = \
                validate(lkvolearner, opt.val_data_root_path, epoch, opt.vis_dir)
            with open(test_csv, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'abs_rel': abs_rel, 'sq_rel': sq_rel, 'rms': rms,
                    'log_rms':log_rms, 'd1_all':d1_all,
                    'a1': a1, 'a2': a2, 'a3': a3})
            if epoch%3==0 and epoch>0:
                pre_train_dir = os.path.join(opt.vis_dir, "train_%02d"%(epoch-1))
                pre_val_dir = os.path.join(opt.vis_dir, "val_%02d"%(epoch-1))
                del_list = [pre_train_dir, pre_val_dir]
                pre_train_dir = os.path.join(opt.vis_dir, "train_%02d"%(epoch-2))
                pre_val_dir = os.path.join(opt.vis_dir, "val_%02d"%(epoch-2))
                del_list.append(pre_train_dir)
                del_list.append(pre_val_dir)
                deldirs(del_list)

    link.finalize()

if __name__=='__main__':
    main()
