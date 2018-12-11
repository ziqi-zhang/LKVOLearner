from DirectVOLayer import DirectVO
from networks import VggDepthEstimator
from ImagePyramid import ImagePyramidLayer
import torch.nn as nn
import torch
import numpy as np
from pdb import set_trace as st
Variable = torch.autograd.Variable

from timeit import default_timer as timer

class FlipLR(nn.Module):
    def __init__(self, imW, dim_w):
        super(FlipLR, self).__init__()
        inv_indices = torch.arange(imW-1, -1, -1).long()
        self.register_buffer('inv_indices', inv_indices)
        self.dim_w = dim_w


    def forward(self, input):
        return input.index_select(self.dim_w, Variable(self.inv_indices))



class LKVOLearner(nn.Module):
    def __init__(self, img_size=[128, 416], ref_frame_idx=1, lambda_S=.5, lambda_K=1, use_ssim=True, smooth_term = 'lap', gpu_ids=[0]):
        super(LKVOLearner, self).__init__()
        self.lkvo = nn.DataParallel(LKVOKernel(img_size, smooth_term = smooth_term), device_ids=gpu_ids)
        self.ref_frame_idx = ref_frame_idx
        self.lambda_S = lambda_S
        self.lambda_K = lambda_K
        self.use_ssim = use_ssim

    def forward(self, frames, camparams, kpts, max_lk_iter_num=10):
        cost, photometric_cost, smoothness_cost, kpts_cost, ref_frame, ref_inv_depth \
            = self.lkvo.forward(frames, camparams, kpts, self.ref_frame_idx, self.lambda_S, self.lambda_K, max_lk_iter_num=max_lk_iter_num, use_ssim=self.use_ssim)
        return cost.mean(), photometric_cost.mean(), smoothness_cost.mean(), kpts_cost.mean(), ref_frame, ref_inv_depth

    def save_model(self, file_path):
        torch.save(self.cpu().lkvo.module.depth_net.state_dict(),
            file_path)
        self.cuda()

    def load_model(self, file_path):
        self.lkvo.module.depth_net.load_state_dict(torch.load(file_path))

    def init_weights(self):
        self.lkvo.module.depth_net.init_weights()

    def get_parameters(self):
        return self.lkvo.module.depth_net.parameters()



class LKVOKernel(nn.Module):
    """
     only support single training isinstance
    """
    def __init__(self, img_size=[128, 416], smooth_term = 'lap'):
        super(LKVOKernel, self).__init__()
        self.img_size = img_size
        self.fliplr_func = FlipLR(imW=img_size[1], dim_w=3)
        self.vo = DirectVO(imH=img_size[0], imW=img_size[1], pyramid_layer_num=5)
        self.depth_net = VggDepthEstimator(img_size)
        self.pyramid_func = ImagePyramidLayer(chan=1, pyramid_layer_num=5)
        self.smooth_term = smooth_term

    def compute_kpts_diff(self, inv_depth_norm, kpts, trans_batch):
        # only support 3 frames
        ref_num, _, kpt_num = kpts.shape[:3]
        assert ref_num == trans_batch.shape[0]
        inv_depth_norm = inv_depth_norm.unsqueeze(0)
        inv_depth_batch = torch.cat((inv_depth_norm[:,0], inv_depth_norm[:,1], inv_depth_norm[:,2], inv_depth_norm[:,1]))
        # src1, ref1, src2, ref2
        kpts = torch.reshape(kpts, (ref_num*2, kpt_num,2)).long()
        kpts_depth = []
        for i in range(ref_num*2):

            depth = inv_depth_batch[i]
            kpt = kpts[i]
            col_select = torch.index_select(depth, 1, kpt[:,0])
            row_select = torch.index_select(col_select, 0, kpt[:,1])
            kpt_depth = row_select.diag()
            # st()
            kpts_depth.append(kpt_depth)
        kpts_depth = torch.stack(kpts_depth)
        for i in range(ref_num):
            kpts_depth[i*2,:]-=trans_batch[i,2]
        diff = []
        for i in range(ref_num):
            diff.append((kpts_depth[i*2,:]-kpts_depth[i*2+1,:]).abs())
        diff = torch.stack(diff)
        print(diff)
        diff_mean = diff.mean()

        return diff_mean

    def check_kpts_diff(self, inv_depth_norm, kpts, trans_batch):
        ref_num, _, kpt_num = kpts.shape[:3]
        assert ref_num == trans_batch.shape[0]
        inv_depth_norm = inv_depth_norm.unsqueeze(0)
        inv_depth_batch = torch.cat((inv_depth_norm[:,0], inv_depth_norm[:,1], inv_depth_norm[:,2], inv_depth_norm[:,1]))
        # src1, ref1, src2, ref2
        kpts = torch.reshape(kpts, (ref_num*2, kpt_num,2)).long()
        kpts_depth = []
        for i in range(ref_num):
            depth_src = inv_depth_batch[2*i]
            kpt_src = kpts[2*i]
            depth_ref = inv_depth_batch[2*i+1]
            kpt_ref = kpts[2*i+1]
            kpt_diff = []
            for j in range(kpt_num):
                u_src, v_src = kpt_src[j]
                u_ref, v_ref = kpt_ref[j]
                d_src = depth_src[v_src, u_src]
                d_ref = depth_ref[v_ref, u_ref]
                d_src-=trans_batch[i,2]
                kpt_diff.append(( d_src - d_ref ).abs())
            kpt_diff = torch.stack(kpt_diff)
            kpts_depth.append(kpt_diff)
        kpts_depth = torch.stack(kpts_depth)
        print(kpts_depth)
        mean = kpts_depth.mean()
        return mean

    def forward(self, frames, camparams, kpts, ref_frame_idx, lambda_S=.5, lambda_K=1, do_data_augment=True, use_ssim=True, max_lk_iter_num=10):
        assert(frames.size(0) == 1 and frames.dim() == 5)
        frames = frames.squeeze(0)
        camparams = camparams.squeeze(0).data
        kpts = kpts.squeeze(0)


        if do_data_augment:
            if np.random.rand()>.5:
                # print("fliplr")
                frames = self.fliplr_func(frames)
                camparams[2] = self.img_size[1] - camparams[2]
                # camparams[5] = self.img_size[0] - camparams[5]

        bundle_size = frames.size(0)
        src_frame_idx = tuple(range(0,ref_frame_idx)) + tuple(range(ref_frame_idx+1,bundle_size))
        # ref_frame = frames[ref_frame_idx, :, :, :]
        # src_frames = frames[src_frame_idx, :, :, :]
        frames_pyramid = self.vo.pyramid_func(frames)
        ref_frame_pyramid = [frame[ref_frame_idx, :, :, :] for frame in frames_pyramid]
        src_frames_pyramid = [frame[src_frame_idx, :, :, :] for frame in frames_pyramid]


        self.vo.setCamera(fx=camparams[0], cx=camparams[2],
                            fy=camparams[4], cy=camparams[5])

        inv_depth_pyramid = self.depth_net.forward((frames-127)/127)
        inv_depth_mean_ten = inv_depth_pyramid[0].mean()*0.1
        #
        # inv_depth0_pyramid = self.pyramid_func(inv_depth_pyramid[0], do_detach=False)
        # ref_inv_depth_pyramid = [depth[ref_frame_idx, :, :] for depth in inv_depth_pyramid]
        # ref_inv_depth0_pyramid = [depth[ref_frame_idx, :, :] for depth in inv_depth0_pyramid]
        # src_inv_depth_pyramid = [depth[src_frame_idx, :, :] for depth in inv_depth_pyramid]
        # src_inv_depth0_pyramid = [depth[src_frame_idx, :, :] for depth in inv_depth0_pyramid]

        inv_depth_norm_pyramid = [depth/inv_depth_mean_ten for depth in inv_depth_pyramid]
        inv_depth0_pyramid = self.pyramid_func(inv_depth_norm_pyramid[0], do_detach=False)
        ref_inv_depth_pyramid = [depth[ref_frame_idx, :, :] for depth in inv_depth_norm_pyramid]
        ref_inv_depth0_pyramid = [depth[ref_frame_idx, :, :] for depth in inv_depth0_pyramid]
        src_inv_depth_pyramid = [depth[src_frame_idx, :, :] for depth in inv_depth_norm_pyramid]
        src_inv_depth0_pyramid = [depth[src_frame_idx, :, :] for depth in inv_depth0_pyramid]

        rot_mat_batch, trans_batch = \
            self.vo.forward(ref_frame_pyramid, src_frames_pyramid, ref_inv_depth0_pyramid, max_itr_num=max_lk_iter_num)
        #
        # smoothness_cost = self.vo.multi_scale_smoothness_cost(inv_depth_pyramid)
        # smoothness_cost += self.vo.multi_scale_smoothness_cost(inv_depth0_pyramid)

        # smoothness_cost = self.vo.multi_scale_smoothness_cost(inv_depth_pyramid, levels=range(1,5))
        # smoothness_cost = self.vo.multi_scale_smoothness_cost(inv_depth0_pyramid, levels=range(1,5))
        photometric_cost = self.vo.compute_phtometric_loss(self.vo.ref_frame_pyramid, src_frames_pyramid, ref_inv_depth_pyramid, src_inv_depth_pyramid, rot_mat_batch, trans_batch, levels=[0,1,2,3], use_ssim=use_ssim)
        smoothness_cost = self.vo.multi_scale_image_aware_smoothness_cost(inv_depth0_pyramid, frames_pyramid, levels=[2,3], type=self.smooth_term) \
                            + self.vo.multi_scale_image_aware_smoothness_cost(inv_depth_norm_pyramid, frames_pyramid, levels=[2,3], type=self.smooth_term)

        kpts_cost = self.compute_kpts_diff(inv_depth_norm_pyramid[0], kpts, trans_batch)
        # kpts_cost_check = self.check_kpts_diff(inv_depth_norm_pyramid[0], kpts, trans_batch)
        # print(kpts_cost)
        # print(kpts_cost_check)
        # st()
        # photometric_cost0, reproj_cost0, _, _ = self.vo.compute_phtometric_loss(self.vo.ref_frame_pyramid, src_frames_pyramid, ref_inv_depth0_pyramid, src_inv_depth0_pyramid, rot_mat_batch, trans_batch)


        # cost = photometric_cost + photometric_cost0 + reproj_cost + reproj_cost0 + lambda_S*smoothness_cost

        cost = photometric_cost + lambda_S*smoothness_cost + lambda_K*kpts_cost
        return cost, photometric_cost, smoothness_cost, kpts_cost, self.vo.ref_frame_pyramid[0], ref_inv_depth0_pyramid[0]*inv_depth_mean_ten


if __name__  == "__main__":
    from KITTIdataset import KITTIdataset
    from torch.utils.data import DataLoader
    from torch import optim
    from torch.autograd import Variable

    dataset = KITTIdataset()
    dataloader = DataLoader(dataset, batch_size=3,
                            shuffle=True, num_workers=4, pin_memory=True)
    lkvolearner = LKVOLearner(gpu_ids = [0, 1, 2])
    def weights_init(m):
        classname = m.__class__.__name__
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            # m.weight.data.normal_(0.0, 0.02)
            m.bias.data = torch.zeros(m.bias.data.size())

    lkvolearner.apply(weights_init)
    lkvolearner.cuda()

    optimizer = optim.Adam(lkvolearner.parameters(), lr=.0001)
    for ii, data in enumerate(dataloader):
        t = timer()
        optimizer.zero_grad()
        frames = Variable(data[0].float().cuda())
        # print(data[1])
        camparams = Variable(data[1])
        a = lkvolearner.forward(frames, camparams)
        print(timer()-t)
        # print(a)
