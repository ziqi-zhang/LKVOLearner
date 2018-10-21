import torch
from torch import optim
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
import scipy.io as sio
import os

from SfMLearner import SfMLearner
from KITTIdataset import KITTIdataset
# from KCSdataset import KCSdataset

from collections import OrderedDict
from options.train_options import TrainOptions
# from util.visualizer import Visualizer

from timeit import default_timer as timer
# data_root_path = '/newfoundland/chaoyang/SfMLearner/data_kitti'

from pdb import set_trace
from util.logger import Logger
from util.util import *
from PIL import Image



def vis_depthmap(input):
    x = (input-input.min()) * (255/(input.max()-input.min()+.00001))
    return x.unsqueeze(2).repeat(1, 1, 3)

def main():
    opt = TrainOptions().parse()
    logger = Logger(opt.tf_log_dir)

    img_size = [opt.imH, opt.imW]

    # visualizer = Visualizer(opt)

    dataset = KITTIdataset(data_root_path=opt.dataroot, img_size=img_size, bundle_size=3)
    # dataset = KCSdataset(img_size=img_size, bundle_size=3)
    dataloader = DataLoader(dataset, batch_size=opt.batchSize,
                            shuffle=True, num_workers=opt.nThreads, pin_memory=True)

    gpu_ids = list(range(opt.batchSize))


    sfmlearner = SfMLearner(img_size=img_size, ref_frame_idx=1, lambda_S=opt.lambda_S, smooth_term = opt.smooth_term, use_ssim=opt.use_ssim)
    sfmlearner.init_weights()


    if opt.which_epoch >= 0:
        print("load pretrained model")
        sfmlearner.load_model(os.path.join(opt.checkpoints_dir, '%s' % (opt.which_epoch)))

    optimizer = optim.Adam(sfmlearner.get_parameters(), lr=.0001)



    sfmlearner.cuda()


    ref_frame_idx = 1



    step_num = 0



    for epoch in range(max(0, opt.which_epoch), opt.epoch_num+1):
        t = timer()
        for ii, data in enumerate(dataloader):
            optimizer.zero_grad()
            frames = Variable(data[0].float().cuda())
            camparams = Variable(data[1])
            cost, photometric_cost, smoothness_cost, frames, inv_depths, _ = \
                sfmlearner.forward(frames, camparams)
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
                print("Print: photometric_cost {:.3f}, smoothness_cost {:.3f}, cost {:.3f}".format(print_photo,
                        print_smooth, print_cost))

                # visualizer.plot_current_errors(step_num, 1, opt,
                #             OrderedDict([('photometric_cost', photometric_cost.data.cpu()[0]),
                #              ('smoothness_cost', smoothness_cost.data.cpu()[0]),
                #              ('cost', cost.data.cpu()[0])]))
                logger.add_scalar('train/photo', print_photo, step_num)
                logger.add_scalar('train/smooth',print_smooth, step_num)
                logger.add_scalar('train/cost', print_cost, step_num)

            if np.mod(step_num, opt.display_freq)==0:
                # frame_vis = frames.data[:,1,:,:,:].permute(0,2,3,1).contiguous().view(-1,opt.imW, 3).cpu().numpy().astype(np.uint8)
                # depth_vis = vis_depthmap(inv_depths.data[:,1,:,:].contiguous().view(-1,opt.imW).cpu()).numpy().astype(np.uint8)
                frame_vis = frames.data.permute(1,2,0).contiguous().cpu().numpy().astype(np.uint8)
                depth_vis = vis_depthmap(inv_depths.data.cpu()).numpy().astype(np.uint8)
                # print("Display: photometric_cost {:.3f}, smoothness_cost {:.3f}, cost {:.3f}".format(photometric_cost.data.cpu()[0],
                #         smoothness_cost.data.cpu()[0], cost.data.cpu()[0]))
                # visualizer.display_current_results(
                #                 OrderedDict([('%s frame' % (opt.name), frame_vis),
                #                         ('%s inv_depth' % (opt.name), depth_vis)]),
                #                         epoch)
                # result_vis = np.hstack([frame_vis, depth_vis])
                # save_image(result_vis, os.path.join(opt.vis_dir, 'depth_%s.png'%step_num))
                # sio.savemat(os.path.join(opt.checkpoints_dir, 'depth_%s.mat' % (step_num)),
                #     {'D': inv_depths.data.cpu().numpy(),
                #      'I': frame_vis})

            if np.mod(step_num, opt.save_latest_freq)==0:
                print("cache model....")
                sfmlearner.save_model(os.path.join(opt.checkpoints_dir, '%s' % (epoch)))
                sfmlearner.cuda()
                print('..... saved')

            if np.mod(step_num, opt.print_freq)==0:
                print()

if __name__=='__main__':
    main()
