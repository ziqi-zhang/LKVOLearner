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
fieldnames = ['abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all',
                'a1', 'a2', 'a3']



def validate(lkvolearner, dataset_root, epoch, vis_dir=None,
                img_size=[128, 416]):
    vgg_depth_net = lkvolearner.lkvo.module.depth_net
    test_file_list = os.path.join(dataset_root, 'list', 'eigen_test_files.txt')
    print("Predicting validate set")
    pred_depths = predKITTI(vgg_depth_net, dataset_root, test_file_list, img_size,
                vis_dir=vis_dir, use_pp=True)

    # pred_depths = np.zeros((697, 128, 375))+1
    print("Evaluating")
    abs_rel, sq_rel, rms, log_rms, d1_all, a1, a2, a3 = \
        evaluate(pred_depths, test_file_list, dataset_root)
    return abs_rel, sq_rel, rms, log_rms, d1_all, a1, a2, a3

def main():
    opt = TrainOptions().parse()
    logger = Logger(opt.tf_log_dir)
    img_size = [opt.imH, opt.imW]

    test_csv = os.path.join(opt.checkpoints_dir, 'test.csv')
    with open(test_csv, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    # visualizer = Visualizer(opt)

    dataset = KITTIdataset(data_root_path=opt.dataroot, img_size=img_size, bundle_size=3)
    dataloader = DataLoader(dataset, batch_size=opt.batchSize,
                            shuffle=True, num_workers=opt.nThreads, pin_memory=True)

    gpu_ids = list(range(opt.batchSize))


    lkvolearner = LKVOLearner(img_size=img_size, ref_frame_idx=1, lambda_S=opt.lambda_S,
            lambda_K = opt.lambda_K, gpu_ids = gpu_ids, smooth_term = opt.smooth_term, use_ssim=opt.use_ssim)
    lkvolearner.init_weights()


    if opt.which_epoch >= 0:
        print("load pretrained model")
        lkvolearner.load_model(os.path.join(opt.checkpoints_dir, '%s_model.pth' % (opt.which_epoch)))

    lkvolearner.cuda()

    ref_frame_idx = 1

    optimizer = optim.Adam(lkvolearner.get_parameters(), lr=.0001)

    step_num = 0



    for epoch in range(max(0, opt.which_epoch), opt.epoch_num+1):
        vis_dir = os.path.join(opt.vis_dir, str(epoch))
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        t = timer()
        for ii, data in enumerate(dataloader):

            optimizer.zero_grad()
            frames = Variable(data[0].float().cuda())
            camparams = Variable(data[1])
            kpts = Variable(data[2]).cuda()
            cost, photometric_cost, smoothness_cost, kpts_cost, frames, inv_depths = \
                lkvolearner.forward(frames, camparams, kpts, max_lk_iter_num=opt.max_lk_iter_num)
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
            optimizer.step()

            step_num+=1

            if np.mod(step_num, opt.print_freq)==0:
                elapsed_time = timer()-t
                print('epoch %s[%s/%s], ... elapsed time: %f (s)' % (epoch, step_num, int(len(dataset)/opt.batchSize), elapsed_time))
                print(inv_depths_mean)
                t = timer()
                print("Print: photometric_cost {:.3f}, smoothness_cost {:.3f}, cost {:.3f}".format(photometric_cost.data.cpu()[0],
                        smoothness_cost.data.cpu()[0], cost.data.cpu()[0]))
                # visualizer.plot_current_errors(step_num, 1, opt,
                #             OrderedDict([('photometric_cost', photometric_cost.data.cpu()[0]),
                #              ('smoothness_cost', smoothness_cost.data.cpu()[0]),
                #              ('cost', cost.data.cpu()[0])]))

                logger.add_scalar('train/photo', photometric_cost.data.cpu()[0], step_num)
                logger.add_scalar('train/smooth',smoothness_cost.data.cpu()[0], step_num)
                logger.add_scalar('train/kpt', kpts_cost.cpu()[0], step_num)
                logger.add_scalar('train/cost', cost.data.cpu()[0], step_num)

            if np.mod(step_num, opt.display_freq)==0:
                # frame_vis = frames.data[:,1,:,:,:].permute(0,2,3,1).contiguous().view(-1,opt.imW, 3).cpu().numpy().astype(np.uint8)
                # depth_vis = vis_depthmap(inv_depths.data[:,1,:,:].contiguous().view(-1,opt.imW).cpu()).numpy().astype(np.uint8)
                frame_vis = frames.data.permute(1,2,0).contiguous().cpu().numpy().astype(np.uint8)
                depth_vis = vis_depthmap(inv_depths.data.cpu().numpy())*255
                depth_vis = depth_vis.astype(np.uint8)
                print("Display: photometric_cost {:.3f}, smoothness_cost {:.3f}, cost {:.3f}".format(photometric_cost.data.cpu()[0],
                        smoothness_cost.data.cpu()[0], cost.data.cpu()[0]))
                # visualizer.display_current_results(
                #                 OrderedDict([('%s frame' % (opt.name), frame_vis),
                #                         ('%s inv_depth' % (opt.name), depth_vis)]),
                #                         epoch)
                result_vis = np.hstack([frame_vis, depth_vis])
                save_image(result_vis, os.path.join(vis_dir, 'depth_%s.png'%step_num))
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

        #eval
        abs_rel, sq_rel, rms, log_rms, d1_all, a1, a2, a3 = \
            validate(lkvolearner, opt.val_data_root_path, epoch)
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'abs_rel': abs_rel, 'sq_rel': sq_rel, 'rms': rms,
                'log_rms':log_rms, 'd1_all':d1_all,
                'a1': a1, 'a2': a2, 'a3': a3})

if __name__=='__main__':
    main()
