from DirectVOLayer import DirectVO
from DirectVOLayerOld import DirectVO as DirectVOOld
from networks import VggDepthEstimator, PoseNet
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

    def forward(self, frames, camparams, kpts, max_lk_iter_num=10, lk_level=1):
        cost, photometric_cost, smoothness_cost, kpts_cost, ref_inv_depth, \
            frame_save, depth_save, warp_img_save \
            = self.lkvo.forward(frames, camparams, kpts, self.ref_frame_idx, \
                                self.lambda_S, self.lambda_K, max_lk_iter_num=max_lk_iter_num, \
                                use_ssim=self.use_ssim, lk_level=lk_level)
        return cost.mean(), photometric_cost.mean(), smoothness_cost.mean(), \
                kpts_cost.mean(), ref_inv_depth, \
                frame_save, depth_save, warp_img_save

    def save_model(self, file_path):
        torch.save(self.cpu().lkvo.module.depth_net.state_dict(),
            file_path)
        self.cuda()

    def load_model(self, depth_net_file_path, pose_net_file_path):
        self.lkvo.module.depth_net.load_state_dict(torch.load(depth_net_file_path))
        self.lkvo.module.pose_net.load_state_dict(torch.load(pose_net_file_path))

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
        self.old_vo = DirectVOOld(imH=img_size[0], imW=img_size[1], pyramid_layer_num=5)
        self.pose_net = PoseNet(3)
        self.depth_net = VggDepthEstimator(img_size)
        self.pyramid_func = ImagePyramidLayer(chan=1, pyramid_layer_num=5)
        self.smooth_term = smooth_term

    def compute_kpts_diff(self, inv_depth_norm, kpts, trans_batch):
        # only support 3 frames
        b, ref_num, _, kpt_num = kpts.shape[:4]
        h, w = inv_depth_norm.shape[-2:]
        inv_depth_batch = torch.cat((inv_depth_norm[:,0:1], inv_depth_norm[:,1:2], inv_depth_norm[:,2:3], inv_depth_norm[:,1:2]), dim=1)
        # src1, ref1, src2, ref2
        kpts = torch.reshape(kpts, (b, ref_num*2, kpt_num,2)).long()
        inv_depth_batch = inv_depth_batch.view(b*ref_num*2, h, w)
        kpts = kpts.view(b*ref_num*2, kpt_num, 2)
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
        trans_batch = trans_batch.view(b*ref_num, 3)
        for i in range(ref_num):
            kpts_depth[i*2,:]-=trans_batch[i,2]
        diff = []
        for i in range(ref_num):
            diff.append((kpts_depth[i*2,:]-kpts_depth[i*2+1,:]).abs())
        diff = torch.stack(diff)
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
        mean = kpts_depth.mean()
        return mean

    def forward(self, frames, camparams, kpts, ref_frame_idx, lambda_S=.5, lambda_K=1, do_data_augment=True, use_ssim=True, max_lk_iter_num=10, lk_level=1):
        # assert(frames.size(0) == 1 and frames.dim() == 5)
        b, bundle_size, c, h, w  = frames.shape

        # frames = frames.squeeze(0)
        # camparams = camparams.squeeze(0).data
        # kpts = kpts.squeeze(0)
        camparams = camparams.data

        # b*bundle, c, h, w
        frames = frames.view(b*bundle_size, c, h, w)
        if do_data_augment:
            if np.random.rand()>.5:
                # print("fliplr")
                frames = self.fliplr_func(frames)
                camparams[:2] = self.img_size[1] - camparams[:2]
                # camparams[5] = self.img_size[0] - camparams[5]

        inv_depth_pyramid = self.depth_net.forward((frames-127)/127)
        inv_depth_mean_ten = inv_depth_pyramid[0].mean(-1).mean(-1)*0.1

        frames = frames.view(b, bundle_size, c, h, w)

        src_frame_idx = tuple(range(0,ref_frame_idx)) + tuple(range(ref_frame_idx+1,bundle_size))
        # ref_frame = frames[ref_frame_idx, :, :, :]
        # src_frames = frames[src_frame_idx, :, :, :]
        frames_cat = frames.view(b*bundle_size, c, h, w)
        frames_pyramid = self.vo.pyramid_func(frames_cat)
        frames_pyramid = [frame.view(b, bundle_size, frame.shape[-3], frame.shape[-2], frame.shape[-1]) for frame in frames_pyramid]

        ref_frame_pyramid = [frame[:,ref_frame_idx, :, :, :] for frame in frames_pyramid]
        src_frames_pyramid = [frame[:,src_frame_idx, :, :, :] for frame in frames_pyramid]

        #
        # inv_depth0_pyramid = self.pyramid_func(inv_depth_pyramid[0], do_detach=False)
        # ref_inv_depth_pyramid = [depth[ref_frame_idx, :, :] for depth in inv_depth_pyramid]
        # ref_inv_depth0_pyramid = [depth[ref_frame_idx, :, :] for depth in inv_depth0_pyramid]
        # src_inv_depth_pyramid = [depth[src_frame_idx, :, :] for depth in inv_depth_pyramid]
        # src_inv_depth0_pyramid = [depth[src_frame_idx, :, :] for depth in inv_depth0_pyramid]
        inv_depth_norm_pyramid = [depth/inv_depth_mean_ten.unsqueeze(-1).unsqueeze(-1) for depth in inv_depth_pyramid]
        inv_depth0_pyramid = self.pyramid_func(inv_depth_norm_pyramid[0], do_detach=False)

        inv_depth_norm_pyramid = [depth.view(b, bundle_size, depth.shape[-2], depth.shape[-1]) for depth in inv_depth_norm_pyramid]
        inv_depth0_pyramid = [depth.view(b, bundle_size, depth.shape[-2], depth.shape[-1]) for depth in inv_depth0_pyramid]
        ref_inv_depth_pyramid = [depth[:, ref_frame_idx, :, :] for depth in inv_depth_norm_pyramid]
        ref_inv_depth0_pyramid = [depth[:, ref_frame_idx, :, :] for depth in inv_depth0_pyramid]
        src_inv_depth_pyramid = [depth[:, src_frame_idx, :, :] for depth in inv_depth_norm_pyramid]
        src_inv_depth0_pyramid = [depth[:, src_frame_idx, :, :] for depth in inv_depth0_pyramid]

        self.vo.setCamera(fx=camparams[:,0:1], cx=camparams[:,2:3],
                            fy=camparams[:,4:5], cy=camparams[:,5:6])
        self.vo.init(ref_frame_pyramid=ref_frame_pyramid, inv_depth_pyramid=ref_inv_depth0_pyramid)
        # b_invH = self.vo.invH_pyramid

        p = self.pose_net.forward((frames.view(b, -1, frames.size(3), frames.size(4))-127) / 127)

        # compute p 1 by 1
        # p_batch = []
        # for b_idx in range(b):
        #     single_frames = frames[b_idx,:,:,:]
        #     p = self.pose_net.forward((single_frames.view(1, -1, single_frames.size(2), single_frames.size(3))-127) / 127)
        #     p_batch.append(p)
        # p =  torch.stack(p_batch).contiguous().squeeze(1)

        rot_mat = self.vo.twist2mat_batch_func(p[:,:,0:3]).contiguous()
        trans = p[:,:,3:6].contiguous()
        batch_rot_mat, batch_trans = self.vo.update_with_init_pose(src_frames_pyramid[0:lk_level], \
                            max_itr_num=max_lk_iter_num, rot_mat_batch=rot_mat, trans_batch=trans)

        for b_idx in range(b):
            single_ref_frame_pyramid = [frame[b_idx,:,:,:] for frame in ref_frame_pyramid]
            single_ref_inv_depth0_pyramid = [depth[b_idx,:,:] for depth in ref_inv_depth0_pyramid]
            single_ref_inv_depth_pyramid = [depth[b_idx,:,:] for depth in ref_inv_depth_pyramid]
            single_frames = frames[b_idx]
            single_src_frames_pyramid = [frame[b_idx] for frame in src_frames_pyramid]
            single_inv_depth0_pyramid = [depth[b_idx,:,:] for depth in inv_depth0_pyramid]
            single_inv_depth_norm_pyramid = [depth[b_idx,:,:] for depth in inv_depth_norm_pyramid]
            single_src_inv_depth_pyramid = [depth[b_idx,:,:] for depth in src_inv_depth_pyramid]
            single_src_inv_depth0_pyramid = [depth[b_idx,:,:] for depth in src_inv_depth0_pyramid]
            single_frames_pyramid = [frame[b_idx,:,:,:] for frame in frames_pyramid]
            single_kpts = kpts[b_idx]

            # predict with posenet
            self.old_vo.setCamera(fx=camparams[b_idx][0], cx=camparams[b_idx][2],
                                fy=camparams[b_idx][4], cy=camparams[b_idx][5])
            self.old_vo.init(ref_frame_pyramid=single_ref_frame_pyramid, inv_depth_pyramid=single_ref_inv_depth0_pyramid)
            p = self.pose_net.forward((single_frames.view(1, -1, single_frames.size(2), single_frames.size(3))-127) / 127)
            # single_invH = self.old_vo.invH_pyramid
            # if b_idx==0:
            #     s_invH = [f.unsqueeze(0) for f in single_invH]
            # else:
            #     s_invH = [torch.cat((b_f, s_f.unsqueeze(0))) for b_f, s_f in zip(s_invH, single_invH)]

            rot_mat_single = self.old_vo.twist2mat_batch_func(p[0,:,0:3]).contiguous()
            trans_single = p[0,:,3:6].contiguous()#*inv_depth_mean_ten
            rot_mat_single, trans_single = self.old_vo.update_with_init_pose(single_src_frames_pyramid[0:lk_level], \
                                max_itr_num=max_lk_iter_num, rot_mat_batch=rot_mat_single, trans_batch=trans_single)

            trans_single = trans_single.unsqueeze(0)
            rot_mat_single = rot_mat_single.unsqueeze(0)
            single_photometric_cost, _ = self.old_vo.compute_phtometric_loss(self.old_vo.ref_frame_pyramid, single_src_frames_pyramid, single_ref_inv_depth_pyramid, \
                                                                single_src_inv_depth_pyramid, rot_mat_single, trans_single, levels=[0,1,2,3], use_ssim=use_ssim)
            if b_idx==0:
                rot_mat_batch = rot_mat_single
                trans_batch = trans_single
            else:
                rot_mat_batch = torch.cat((rot_mat_batch, rot_mat_single))
                trans_batch = torch.cat((trans_batch, trans_single))


            if b_idx==0:
                batch_photometric_cost = [single_photometric_cost]
            else:
                batch_photometric_cost.append(single_photometric_cost)

            # predict with only ddvo
            # rot_mat_batch, trans_batch = \
            #     self.vo.forward(ref_frame_pyramid, src_frames_pyramid, ref_inv_depth0_pyramid, max_itr_num=max_lk_iter_num)

            #
            # smoothness_cost = self.vo.multi_scale_smoothness_cost(inv_depth_pyramid)
            # smoothness_cost += self.vo.multi_scale_smoothness_cost(inv_depth0_pyramid)

            # smoothness_cost = self.vo.multi_scale_smoothness_cost(inv_depth_pyramid, levels=range(1,5))
            # smoothness_cost = self.vo.multi_scale_smoothness_cost(inv_depth0_pyramid, levels=range(1,5))

        if False:
            # check invH
            invH_sum = (b_invH[0]-s_invH[0]).abs().sum()
            for i in range(1, 5):
                invH_sum += (b_invH[i]-s_invH[i]).abs().sum()
            print("invH error: {}".format(invH_sum))

        batch_photometric_cost = torch.stack(batch_photometric_cost)



        photometric_cost, warp_img_save = self.vo.compute_phtometric_loss(ref_frame_pyramid, src_frames_pyramid, ref_inv_depth_pyramid, \
                                                                src_inv_depth_pyramid, batch_rot_mat, batch_trans, levels=[0,1,2,3], use_ssim=use_ssim)

        smoothness_cost = self.vo.multi_scale_image_aware_smoothness_cost(inv_depth0_pyramid, frames_pyramid, levels=[2,3], type=self.smooth_term) \
                            + self.vo.multi_scale_image_aware_smoothness_cost(inv_depth_norm_pyramid, frames_pyramid, levels=[2,3], type=self.smooth_term)
        kpts_cost = self.compute_kpts_diff(inv_depth_norm_pyramid[0], kpts, batch_trans)
        # kpts_cost_check = self.check_kpts_diff(inv_depth_norm_pyramid[0], kpts, trans_batch)
        # print(kpts_cost)
        # print(kpts_cost_check)
        # st()
        # photometric_cost0, reproj_cost0, _, _ = self.vo.compute_phtometric_loss(self.vo.ref_frame_pyramid, src_frames_pyramid, ref_inv_depth0_pyramid, src_inv_depth0_pyramid, rot_mat_batch, trans_batch)

        print("rot: {}, {:.5f}%".format((batch_rot_mat-rot_mat_batch).abs().sum(), \
              (batch_rot_mat-rot_mat_batch).abs().sum() / rot_mat_batch.abs().sum() ))
        print("t  : {}, {:.5f}%".format((batch_trans-trans_batch).abs().sum(), \
              (batch_trans-trans_batch).abs().sum() / trans_batch.abs().sum() ))
        photometric_cost_1by1 = batch_photometric_cost.mean()
        print("phot {}".format((photometric_cost - photometric_cost_1by1).abs()))
        st()

        # cost = photometric_cost + photometric_cost0 + reproj_cost + reproj_cost0 + lambda_S*smoothness_cost
        cost = photometric_cost + lambda_S*smoothness_cost + lambda_K*kpts_cost

        frame_save = [frames[0][i] for i in range(3)]
        depth_save = [inv_depth0_pyramid[0][0][i] for i in range(3)]

        return cost, photometric_cost, smoothness_cost, kpts_cost, \
                inv_depth0_pyramid[0].view(b*bundle_size, h, w)*inv_depth_mean_ten.unsqueeze(-1).unsqueeze(-1), \
                frame_save, depth_save, warp_img_save



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
