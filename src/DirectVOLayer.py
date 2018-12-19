# class DirectVO():
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.nn import AvgPool2d
from ImagePyramid import ImagePyramidLayer
from BilinearSampling import grid_bilinear_sampling
from MatInverse import inv, inv_batch
import os
from pdb import set_trace as st

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from timeit import default_timer as timer

IMG_CHAN = 3

# helper functions

def meshgrid(x, y):
    imW = x.size(1)
    imH = y.size(1)
    X = x.unsqueeze(1).repeat(1, imH, 1)
    Y = y.unsqueeze(2).repeat(1, 1, imW)
    return X, Y

def inv_rigid_transformation(rot_mat_batch, trans_batch):
    inv_rot_mat_batch = rot_mat_batch.transpose(1,2)
    inv_trans_batch = -inv_rot_mat_batch.bmm(trans_batch.unsqueeze(-1)).squeeze(-1)
    return inv_rot_mat_batch, inv_trans_batch

class LaplacianLayer(nn.Module):
    def __init__(self):
        super(LaplacianLayer, self).__init__()
        w_nom = torch.FloatTensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).view(1,1,3,3)
        w_den = torch.FloatTensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]]).view(1,1,3,3)
        self.register_buffer('w_nom', w_nom)
        self.register_buffer('w_den', w_den)

    def forward(self, input, do_normalize=True):
        assert(input.dim() == 2 or input.dim()==3 or input.dim()==4)
        input_size = input.size()
        if input.dim()==4:
            x = input.view(input_size[0]*input_size[1], 1,
                            input_size[2], input_size[3])
        elif input.dim()==3:
            x = input.unsqueeze(1)
        else:
            x = input.unsqueeze(0).unsqueeze(0)
        x_nom = torch.nn.functional.conv2d(input=x,
                        weight=Variable(self.w_nom),
                        stride=1,
                        padding=0)
        if do_normalize:
            x_den = torch.nn.functional.conv2d(input=x,
                        weight=Variable(self.w_den),
                        stride=1,
                        padding=0)
            # x_den = x.std() + 1e-5
            x = (x_nom.abs()/x_den)
        else:
            x = x_nom.abs()
        if input.dim() == 4:
            return x.view(input_size[0], input_size[1], input_size[2]-2, input_size[3]-2)
        elif input.dim() == 3:
            return x.squeeze(1)
        elif input.dim() == 2:
            return x.squeeze(0).squeeze(0)



class GradientLayer(nn.Module):
    def __init__(self):
        super(GradientLayer, self).__init__()
        wx = torch.FloatTensor([-.5, 0, .5]).view(1, 1, 1, 3)
        wy = torch.FloatTensor([[-.5], [0], [.5]]).view(1, 1, 3, 1)
        self.register_buffer('wx', wx)
        self.register_buffer('wy', wy)
        self.padx_func = torch.nn.ReplicationPad2d((1,1,0,0))
        self.pady_func = torch.nn.ReplicationPad2d((0,0,1,1))

    def forward(self, img):
        b, k, h, w = img.shape
        # img_ = img.unsqueeze(1)
        img_ = img.contiguous().view(b*k, 1, h, w)
        img_padx = self.padx_func(img_)
        img_dx = torch.nn.functional.conv2d(input=img_padx,
                        weight=Variable(self.wx),
                        stride=1,
                        padding=0)

        img_pady = self.pady_func(img_)
        img_dy = torch.nn.functional.conv2d(input=img_pady,
                        weight=Variable(self.wy),
                        stride=1,
                        padding=0)
        img_dx = img_dx.view(b, k, h, w)
        img_dy = img_dy.view(b, k, h, w)
        return img_dx, img_dy


class Twist2Mat(nn.Module):
    def __init__(self):
        super(Twist2Mat, self).__init__()
        self.register_buffer('o', torch.zeros(1,1,1))
        self.register_buffer('E', torch.eye(3))

    def cprodmat_batch(self, a_batch):
        batch_size, ref_size, _ = a_batch.size()
        o = Variable(self.o).expand(batch_size, ref_size, 1)
        a0 = a_batch[:, :, 0:1]
        a1 = a_batch[:, :, 1:2]
        a2 = a_batch[:, :, 2:3]
        return torch.cat((o, -a2, a1, a2, o, -a0, -a1, a0, o), 2).view(batch_size, ref_size, 3, 3)

    def forward(self, twist_batch):
        batch_size, ref_size, _ = twist_batch.size()
        rot_angle = twist_batch.norm(p=2, dim=2).view(batch_size, ref_size, 1).clamp(min=1e-5)
        rot_axis = twist_batch / rot_angle.expand(batch_size, ref_size, 3)
        A = self.cprodmat_batch(rot_axis)
        A = A.view(batch_size*ref_size, 3, 3)
        res = Variable(self.E).view(1, 3, 3).expand(batch_size*ref_size, 3, 3)\
            + A*rot_angle.sin().view(batch_size*ref_size, 1, 1).expand(batch_size*ref_size, 3, 3)\
            + A.bmm(A)*((1-rot_angle.cos()).view(batch_size*ref_size, 1, 1).expand(batch_size*ref_size, 3, 3))
        res = res.view(batch_size, ref_size, 3, 3)
        return res

def compute_img_stats(img):
    # img_pad = torch.nn.ReplicationPad2d(1)(img)
    img_pad = img
    mu = AvgPool2d(kernel_size=3, stride=1, padding=0)(img_pad)
    sigma = AvgPool2d(kernel_size=3, stride=1, padding=0)(img_pad**2) - mu**2
    return mu, sigma

def compute_img_stats_pyramid(frames_pyramid):
    mu_pyramid = []
    sigma_pyramid = []
    for layer_idx in range(len(frames_pyramid)):
        mu, sigma = compute_img_stats(frames_pyramid[layer_idx])
        mu_pyramid.append(mu)
        sigma_pyramid.append(sigma)
    return mu_pyramid, sigma_pyramid


def compute_SSIM(img0, mu0, sigma0, img1, mu1, sigma1):
    # img0_img1_pad = torch.nn.ReplicationPad2d(1)(img0 * img1)
    img0_img1_pad = img0*img1
    sigma01 = AvgPool2d(kernel_size=3, stride=1, padding=0)(img0_img1_pad) - mu0*mu1
    # C1 = .01 ** 2
    # C2 = .03 ** 2
    C1 = .001
    C2 = .009

    ssim_n = (2*mu0*mu1 + C1) * (2*sigma01 + C2)
    ssim_d = (mu0**2 + mu1**2 + C1) * (sigma0 + sigma1 + C2)
    ssim = ssim_n / ssim_d
    return ((1-ssim)*.5).clamp(0, 1)


def compute_photometric_cost(img_diff, mask):
    # cost = ((img0.expand_as(img1) - img1) * mask).abs().sum()
    k = img_diff.size(1)
    batch_size = img_diff.size(0)
    mask_ = (mask.view(batch_size, 1, -1) * (1/127.5)).expand_as(img_diff)
    # mask_ = (mask.view(batch_size, 1, -1)*(1/127.5)).repeat(1, k, 1)
    cost = img_diff.abs() * mask_
    return cost

def compute_photometric_cost_norm(img_diff, mask):
    # cost = ((img0.expand_as(img1) - img1) * mask).abs().sum()
    # k = img_diff.size(1)
    # batch_size = img_diff.size(0)
    # mask_ = mask.view(batch_size, 1, -1).expand_as(img_diff)
    cost = img_diff.abs().sum(2) * mask
    # cost = img_diff.abs() * mask_
    num_in_view = mask.sum(-1)
    # cost_norm = cost.contiguous().sum() / (num_in_view+1e-10)
    cost_norm = cost.sum(-1) / (num_in_view+1e-10)
    # print(mask.size())
    return cost_norm * (1 / 127.5), (num_in_view / mask.size(2)).min()


def gradient(input, do_normalize=False):
    if input.dim() == 2:
        D_ry = input[1:, :]
        D_ly = input[:-1, :]
        D_rx = input[:, 1:]
        D_lx = input[:, :-1]
    elif input.dim() == 3:
        D_ry = input[:, 1:, :]
        D_ly = input[:, :-1, :]
        D_rx = input[:, :, 1:]
        D_lx = input[:, :, :-1]
    elif input.dim() == 4:
        D_ry = input[:, :, 1:, :]
        D_ly = input[:, :, :-1, :]
        D_rx = input[:, :, :, 1:]
        D_lx = input[:, :, :, :-1]
    # if input.dim() == 2:
    #     D_dy = input[1:, :] - input[:-1, :]
    #     D_dx = input[:, 1:] - input[:, :-1]
    # elif input.dim() == 3:
    #     D_dy = input[:, 1:, :] - input[:, :-1, :]
    #     D_dx = input[:, :, 1:] - input[:, :, :-1]
    # elif input.dim() == 4:
    #     D_dy = input[:, :, 1:, :] - input[:, :, :-1, :]
    #     D_dx = input[:, :, :, 1:] - input[:, :, :, :-1]
    Dx = D_rx - D_lx
    Dy = D_ry - D_ly
    if do_normalize:
        Dx = Dx / (D_rx + D_lx)
        Dy = Dy / (D_ry + D_ly)
    return Dx, Dy




class DirectVO(nn.Module):

    def __init__(self, imH=128, imW=416, pyramid_layer_num=5, max_itr_num=20):
        super(DirectVO, self).__init__()
        self.max_itr_num = max_itr_num
        self.imH = imH
        self.imW = imW
        self.pyramid_layer_num = pyramid_layer_num

        self.twist2mat_batch_func = Twist2Mat()
        self.img_gradient_func = GradientLayer()
        self.pyramid_func = ImagePyramidLayer(chan=3,
                                pyramid_layer_num=self.pyramid_layer_num)
        self.laplacian_func = LaplacianLayer()

        x_pyramid, y_pyramid = self.pyramid_func.get_coords(self.imH, self.imW)


        # self.x_pyramid = nn.ParameterList(
        #             [nn.Parameter(torch.from_numpy(x), False) for x in x_pyramid])
        # self.y_pyramid = nn.ParameterList(
        #             [nn.Parameter(torch.from_numpy(y), False) for y in y_pyramid])

        for i in range(self.pyramid_layer_num):
            self.register_buffer('x_'+str(i), torch.from_numpy(x_pyramid[i]).float())
            self.register_buffer('y_'+str(i), torch.from_numpy(y_pyramid[i]).float())
        self.register_buffer('o', torch.zeros(1,1))
        self.register_buffer('E', torch.eye(3))


    def setCamera(self, cx, cy, fx, fy):
        self.camparams = dict(fx=fx, fy=fy, cx=cx, cy=cy)


    def init(self, ref_frame_pyramid, inv_depth_pyramid):
        # batch_size * ref_frame 3 * H * W
        assert(self.pyramid_layer_num == len(inv_depth_pyramid))
        self.inv_depth_pyramid = inv_depth_pyramid
        # self.ref_frame_pyramid = self.pyramid_func(ref_frame)
        self.ref_frame_pyramid = ref_frame_pyramid

        for i in range(self.pyramid_layer_num):
            assert(self.ref_frame_pyramid[i].size(-1) == inv_depth_pyramid[i].size(-1))
            assert (self.ref_frame_pyramid[i].size(-2) == inv_depth_pyramid[i].size(-2))
        self.init_lk_terms()



    def init_lk_terms(self):
        # self.inv_depth_pyramid, self.x_pyramid, self.y_pyramid = buildImagePyramid(inv_depth, self.pyramid_layer_num)
        self.xy_pyramid = []
        self.ref_imgrad_x_pyramid = []
        self.ref_imgrad_y_pyramid = []
        self.invH_pyramid = []
        self.dIdp_pyramid = []
        self.invH_dIdp_pyramid = []

        for i in range(self.pyramid_layer_num):
            b, _, h, w = self.ref_frame_pyramid[i].size()

            x = getattr(self, 'x_'+str(i))
            y = getattr(self, 'y_'+str(i))

            x = x.unsqueeze(0).repeat(b, 1)
            y = y.unsqueeze(0).repeat(b, 1)
            x = (x - self.camparams['cx']) / self.camparams['fx']
            y = (y - self.camparams['cy']) / self.camparams['fy']

            # old_x = (Variable(getattr(self, 'x_'+str(i))) - self.camparams['cx']) / self.camparams['fx']
            # old_y = (Variable(getattr(self, 'y_'+str(i))) - self.camparams['cy']) / self.camparams['fy']

            X, Y = meshgrid(x, y)
            xy = torch.cat((X.view(b, X[0].numel()).unsqueeze(1),
                            Y.view(b, Y[0].numel()).unsqueeze(1)), 1)
            self.xy_pyramid.append(xy)

            # compute image gradient
            imgrad_x, imgrad_y = self.img_gradient_func(self.ref_frame_pyramid[i])

            fx = (self.camparams['fx']/2**i).view(b, 1, 1, 1)
            fy = (self.camparams['fy']/2**i).view(b, 1, 1, 1)
            self.ref_imgrad_x_pyramid.append(imgrad_x*fx)
            self.ref_imgrad_y_pyramid.append(imgrad_y*fy)

            # precompute terms for LK regress
            dIdp = self.compute_dIdp(self.ref_imgrad_x_pyramid[i],
                                     self.ref_imgrad_y_pyramid[i],
                                     self.inv_depth_pyramid[i],
                                     self.xy_pyramid[i])
            self.dIdp_pyramid.append(dIdp)

            # check error between inv and batch_inv
            # invH = inv(dIdp.t().mm(dIdp))
            # b_invH = inv_batch(dIdp.transpose(1,2).bmm(dIdp))
            # H = dIdp[0].t().mm(dIdp[0])
            # invH = inv(H).unsqueeze(0)
            # for i in range(1, dIdp.shape[0]):
            #     H = dIdp[i].t().mm(dIdp[i])
            #     invH = torch.cat((invH, inv(H).unsqueeze(0)))
            # print("invH error: {}".format((b_invH-invH).abs().sum()))
            invH = inv_batch(dIdp.transpose(1,2).bmm(dIdp))

            self.invH_pyramid.append(invH)

            # check error between inv and batch_inv
            # self.invH_dIdp_pyramid.append(self.invH_pyramid[-1].mm(dIdp.t()))
            self.invH_dIdp_pyramid.append(invH.bmm(dIdp.transpose(1,2)))

    def init_xy_pyramid(self, ref_frames_pyramid):
        # self.inv_depth_pyramid, self.x_pyramid, self.y_pyramid = buildImagePyramid(inv_depth, self.pyramid_layer_num)
        self.xy_pyramid = []
        self.ref_imgrad_x_pyramid = []
        self.ref_imgrad_y_pyramid = []

        for i in range(self.pyramid_layer_num):
            _, h, w = ref_frames_pyramid[i].size()

            x = (Variable(getattr(self, 'x_'+str(i))) - self.camparams['cx']) / self.camparams['fx']
            y = (Variable(getattr(self, 'y_'+str(i))) - self.camparams['cy']) / self.camparams['fy']

            X, Y = meshgrid(x, y)
            xy = torch.cat((X.view(1, X.numel()),
                            Y.view(1, Y.numel())), 0)
            self.xy_pyramid.append(xy)



    def compute_dIdp(self, imgrad_x, imgrad_y, inv_depth, xy):
        b, k, h, w = imgrad_x.size()
        b, _, pt_num = xy.size()
        assert(h*w == pt_num)
        feat_dim = pt_num*k
        x = xy[:, 0, :].view(b, pt_num, 1)
        y = xy[:, 1, :].view(b, pt_num, 1)
        xty = x * y
        O = Variable(self.o).expand(b, pt_num, 1)
        inv_depth_ = inv_depth.view(b, pt_num, 1)

        dxdp = torch.cat((-xty, 1 + x ** 2, -y, inv_depth_, O, -inv_depth_.mul(x)), 2)
        dydp = torch.cat((-1 - y ** 2, xty, x, O, inv_depth_, -inv_depth_.mul(y)), 2)

        imgrad_x_ = imgrad_x.view(b, feat_dim, 1).expand(b, feat_dim, 6)
        imgrad_y_ = imgrad_y.view(b, feat_dim, 1).expand(b, feat_dim, 6)
        dIdp = imgrad_x_.mul(dxdp.repeat(1, k, 1)) + \
            imgrad_y_.mul(dydp.repeat(1, k, 1))
        return dIdp



    def LKregress(self, invH_dIdp, mask, It):
        b, ref_num, pt_num = mask.size()
        _, _, k, _ = It.size()
        feat_dim = k*pt_num
        invH_dIdp_ = invH_dIdp.view(b, 1, 6, feat_dim).expand(b, ref_num, 6, feat_dim)
        mask_ = mask.view(b, ref_num, 1, pt_num).expand(b, ref_num, k, pt_num)
        # huber_weights = ((255*.2) / (It.abs()+1e-5)).clamp(max=1)
        # huber_weights = Variable(huber_weights.data)
        # dp = invH_dIdp_.bmm((mask_* huber_weights * It).view(batch_size, feat_dim, 1))
        invH_dIdp_ = invH_dIdp_.contiguous().view(b*ref_num, 6, feat_dim)
        masked_It = mask_ * It
        masked_It = masked_It.view(b*ref_num, k, pt_num)
        dp = invH_dIdp_.bmm(masked_It.view(b*ref_num, feat_dim, 1))
        return dp.view(b, ref_num, 6)

    def warp_batch(self, img_batch, level_idx, R_batch, t_batch):
        return self.warp_batch_func(img_batch, self.inv_depth_pyramid[level_idx], level_idx, R_batch, t_batch)


    def warp_batch_func(self, img_batch, inv_depth, level_idx, R_batch, t_batch):
        b, ref_size, k, h, w = img_batch.size()
        xy = self.xy_pyramid[level_idx]
        _, _, N = xy.size()
        batch_size = b*ref_size
        R_batch = R_batch.view(batch_size, 3, 3)
        t_batch = t_batch.view(batch_size, 3)
        xy = xy.view(b, 1, 2, N).expand(b, ref_size, 2, N).contiguous().view(batch_size, 2, N)
        # xyz = R_batch.bmm(torch.cat((xy.view(1, 2, N).expand(batch_size, 2, N), Variable(self.load_to_device(torch.ones(batch_size, 1, N)))), 1)) \
        #     + t_batch.view(batch_size, 3, 1).expand(batch_size, 3, N) * inv_depth.view(1, 1, N).expand(batch_size, 3, N)
        batch_inv_depth = inv_depth.view(b, 1, N).expand(b, ref_size, N)
        batch_inv_depth = batch_inv_depth.contiguous().view(batch_size, 1, N).expand(batch_size, 3, N)
        xyz = R_batch[:, :, 0:2].bmm(xy)\
            + R_batch[:, :, 2:3].expand(batch_size, 3, N)\
            + t_batch.contiguous().view(batch_size, 3, 1).expand(batch_size, 3, N) * batch_inv_depth
        z = xyz[:, 2:3, :].clamp(min=1e-10)
        xy_warp = xyz[:, 0:2, :] / z.expand(batch_size, 2, N)
        # u_warp = ((xy_warp[:, 0, :]*self.camparams['fx'] + self.camparams['cx'])/2**level_idx - .5).view(batch_size, N)
        # v_warp = ((xy_warp[:, 1, :]*self.camparams['fy'] + self.camparams['cy'])/2**level_idx - .5).view(batch_size, N)
        # print(self.x_pyramid[level_idx][0])
        xy_warp = xy_warp.view(b, ref_size, 2, N)
        u_warp = ((xy_warp[:, :, 0, :] * self.camparams['fx'].view(b,1,1) + self.camparams['cx'].view(b,1,1)) - \
                    getattr(self, 'x_'+str(level_idx))[0]).view(b, ref_size, N) / 2 ** level_idx
        v_warp = ((xy_warp[:, :, 1, :] * self.camparams['fy'].view(b,1,1) + self.camparams['cy'].view(b,1,1)) - \
                    getattr(self, 'y_'+str(level_idx))[0]).view(b, ref_size, N) / 2 ** level_idx

        Q, in_view_mask =  grid_bilinear_sampling(img_batch, u_warp, v_warp)
        return Q, in_view_mask * (z.view_as(in_view_mask)>1e-10).float()

    def warp_batch_func_photo_error(self, img_batch, inv_depth, level_idx, R_batch, t_batch):
        b, ref_size, k, h, w = img_batch.size()
        xy = self.xy_pyramid[level_idx]
        _, _, N = xy.size()
        batch_size = b*ref_size
        R_batch = R_batch.view(batch_size, 3, 3)
        t_batch = t_batch.view(batch_size, 3)
        xy = xy.view(b, 1, 2, N).expand(b, ref_size, 2, N).contiguous().view(batch_size, 2, N)
        # xyz = R_batch.bmm(torch.cat((xy.view(1, 2, N).expand(batch_size, 2, N), Variable(self.load_to_device(torch.ones(batch_size, 1, N)))), 1)) \
        #     + t_batch.view(batch_size, 3, 1).expand(batch_size, 3, N) * inv_depth.view(1, 1, N).expand(batch_size, 3, N)
        batch_inv_depth = inv_depth.contiguous().view(batch_size, 1, N).expand(batch_size, 3, N)
        xyz = R_batch[:, :, 0:2].bmm(xy)
        xyz += R_batch[:, :, 2:3].expand(batch_size, 3, N)
        xyz += t_batch.contiguous().view(batch_size, 3, 1).expand(batch_size, 3, N) * batch_inv_depth
        xyz = R_batch[:, :, 0:2].bmm(xy)\
            + R_batch[:, :, 2:3].expand(batch_size, 3, N)\
            + t_batch.contiguous().view(batch_size, 3, 1).expand(batch_size, 3, N) * batch_inv_depth
        z = xyz[:, 2:3, :].clamp(min=1e-10)
        xy_warp = xyz[:, 0:2, :] / z.expand(batch_size, 2, N)
        # u_warp = ((xy_warp[:, 0, :]*self.camparams['fx'] + self.camparams['cx'])/2**level_idx - .5).view(batch_size, N)
        # v_warp = ((xy_warp[:, 1, :]*self.camparams['fy'] + self.camparams['cy'])/2**level_idx - .5).view(batch_size, N)
        # print(self.x_pyramid[level_idx][0])
        xy_warp = xy_warp.view(b, ref_size, 2, N)
        u_warp = ((xy_warp[:, :, 0, :] * self.camparams['fx'].view(b,1,1) + self.camparams['cx'].view(b,1,1)) - \
                    getattr(self, 'x_'+str(level_idx))[0]).view(b, ref_size, N) / 2 ** level_idx
        v_warp = ((xy_warp[:, :, 1, :] * self.camparams['fy'].view(b,1,1) + self.camparams['cy'].view(b,1,1)) - \
                    getattr(self, 'y_'+str(level_idx))[0]).view(b, ref_size, N) / 2 ** level_idx

        Q, in_view_mask =  grid_bilinear_sampling(img_batch, u_warp, v_warp)
        return Q, in_view_mask * (z.view_as(in_view_mask)>1e-10).float()



    def compute_phtometric_loss(self, ref_frames_pyramid, src_frames_pyramid, ref_inv_depth_pyramid, src_inv_depth_pyramid,
                                rot_mat_batch, trans_batch,
                                use_ssim=True, levels=None,
                                ref_expl_mask_pyramid=None,
                                src_expl_mask_pyramid=None):
        b = rot_mat_batch.size(0)
        bundle_size = rot_mat_batch.size(1)+1

        rot_mat_batch = rot_mat_batch.view(b*(bundle_size-1), 3, 3)
        trans_batch = trans_batch.view(b*(bundle_size-1), 3)
        inv_rot_mat_batch, inv_trans_batch = inv_rigid_transformation(rot_mat_batch, trans_batch)
        inv_rot_mat_batch = inv_rot_mat_batch.contiguous().view(b, bundle_size-1, 3, 3)
        inv_trans_batch = inv_trans_batch.contiguous().view(b, bundle_size-1, 3)
        rot_mat_batch = rot_mat_batch.view(b, bundle_size-1, 3, 3)
        trans_batch = trans_batch.view(b, bundle_size-1, 3)

        src_pyramid = []
        ref_pyramid = []
        depth_pyramid = []
        if levels is None:
            levels = range(self.pyramid_layer_num)

        use_expl_mask = not (ref_expl_mask_pyramid is None \
                            or src_expl_mask_pyramid is None)
        if use_expl_mask:
            expl_mask_pyramid = []
            for level_idx in levels:
                ref_mask = ref_expl_mask_pyramid[level_idx].unsqueeze(0).repeat(bundle_size-1,1,1)
                src_mask = src_expl_mask_pyramid[level_idx]
                expl_mask_pyramid.append(torch.cat(
                    (ref_mask, src_mask), 0))

        # for level_idx in range(len(ref_frames_pyramid)):
        for level_idx in levels:
        # for level_idx in range(3):
            ref_frame = ref_frames_pyramid[level_idx].unsqueeze(1).repeat(1, bundle_size-1, 1, 1, 1)
            src_frame = src_frames_pyramid[level_idx]
            ref_depth = ref_inv_depth_pyramid[level_idx].unsqueeze(1).repeat(1, bundle_size-1, 1, 1)
            src_depth = src_inv_depth_pyramid[level_idx]
            h, w = ref_frame.shape[-2:]
            # ref_frame = ref_frame.view(b*(bundle_size-1), 3, h, w)
            # src_frame = src_frame.view(b*(bundle_size-1), 3, h, w)
            # ref_depth = ref_depth.view(b*(bundle_size-1), h, w)
            # src_depth = src_depth.view(b*(bundle_size-1), h, w)
            # print(src_depth.size())

            ref_pyramid.append(torch.cat((ref_frame,
                                    src_frame), 1)/127.5)
            src_pyramid.append(torch.cat((src_frame,
                                    ref_frame), 1)/127.5)
            depth_pyramid.append(torch.cat((ref_depth,
                                    src_depth), 1))

        rot_mat = torch.cat((rot_mat_batch,
                            inv_rot_mat_batch) ,1)
        trans = torch.cat((trans_batch,
                           inv_trans_batch), 1)

        loss = 0

        frames_warp_pyramid = []
        ref_frame_warp_pyramid = []

        # for level_idx in range(len(ref_pyramid)):
        # for level_idx in range(3):
        for level_idx in levels:
            # print(depth_pyramid[level_idx].size())
            b, _, h, w = depth_pyramid[level_idx].size()

            warp_img, in_view_mask = self.warp_batch_func_photo_error(
                    src_pyramid[level_idx],
                    depth_pyramid[level_idx],
                    level_idx, rot_mat, trans)

            # warp_img_save is the warped img from src to ref with full size

            warp_img = warp_img.view(b, 2*(bundle_size-1), IMG_CHAN, h, w)
            in_view_mask = in_view_mask.view(b, 2*(bundle_size-1), h, w)

            if level_idx==0:
                split_idx = b*(bundle_size-1)
                warp_img_save = [warp_img[0][2:3], warp_img[0][0:2], warp_img[0][3:4]]
            if use_expl_mask:
                mask = in_view_mask.view(-1,h,w)*expl_mask_pyramid[level_idx]
            else:
                mask = in_view_mask
            mask_expand = mask.view(b, (bundle_size-1)*2, 1, h, w).expand(b, (bundle_size-1)*2, IMG_CHAN, h, w)
            rgb_loss = ((ref_pyramid[level_idx] - warp_img).abs()*mask_expand).mean()
            if use_ssim and level_idx<1:
                # print("compute ssim loss------")
                warp_img_bundle = warp_img.view(b*2*(bundle_size-1), IMG_CHAN, h, w)
                ref_bundle = ref_pyramid[level_idx].view(b*2*(bundle_size-1), IMG_CHAN, h, w)
                mask_bundle = mask_expand.view(b*2*(bundle_size-1), IMG_CHAN, h, w)
                warp_mu, warp_sigma = compute_img_stats(warp_img_bundle)
                ref_mu, ref_sigma = compute_img_stats(ref_bundle)
                ssim = compute_SSIM(ref_bundle,
                                ref_mu,
                                ref_sigma,
                                warp_img_bundle,
                                warp_mu,
                                warp_sigma)
                ssim_loss = (ssim*mask_bundle[:,:,1:-1,1:-1]).mean()
                loss += .85*ssim_loss+.15*rgb_loss
            else:
                loss += rgb_loss

        #     frames_warp_pyramid.append(warp_img*127.5)
        #     ref_frame_warp_pyramid.append(ref_pyramid[level_idx]*127.5)
        #
        # return loss, frames_warp_pyramid, ref_frame_warp_pyramid
        return loss, [img.detach() for img in warp_img_save]

    def compute_smoothness_cost(self, inv_depth):
        x = self.laplacian_func(inv_depth)
        return x.mean()

    def compute_image_aware_laplacian_smoothness_cost(self, depth, img):
        img_lap = self.laplacian_func(img/255, do_normalize=False)
        depth_lap = self.laplacian_func(depth, do_normalize=False)
        x = (-img_lap.mean(1)).exp()*(depth_lap)
        return x.mean()

    def compute_image_aware_2nd_smoothness_cost(self, depth, img):
        img_lap = self.laplacian_func(img/255, do_normalize=False)
        depth_grad_x, depth_grad_y = gradient(depth, do_normalize=False)
        depth_grad_x2, depth_grad_xy = gradient(depth_grad_x, do_normalize=False)
        depth_grad_yx, depth_grad_y2 = gradient(depth_grad_y, do_normalize=False)
        return depth_grad_x2.abs().mean() \
            + depth_grad_xy.abs().mean() + depth_grad_yx.abs().mean() + depth_grad_y2.abs().mean()


    def compute_image_aware_1st_smoothness_cost(self, depth, img):
        depth_grad_x, depth_grad_y = gradient(depth, do_normalize=False)
        img_grad_x, img_grad_y = gradient(img/255, do_normalize=False)
        if img.dim() == 3:
            weight_x = torch.exp(-img_grad_x.abs().mean(0))
            weight_y = torch.exp(-img_grad_y.abs().mean(0))
            cost = ((depth_grad_x.abs() * weight_x)[:-1, :] + (depth_grad_y.abs() * weight_y)[:, :-1]).mean()
        else:
            weight_x = torch.exp(-img_grad_x.abs().mean(1))
            weight_y = torch.exp(-img_grad_y.abs().mean(1))
            cost = ((depth_grad_x.abs() * weight_x)[:, :-1, :] + (depth_grad_y.abs() * weight_y)[:, :, :-1]).mean()
        return cost




    def multi_scale_smoothness_cost(self, inv_depth_pyramid, levels = None):
        cost = 0
        if levels is None:
            levels = range(self.pyramid_layer_num)

        # for level_idx in range(2, self.pyramid_layer_num):
        for level_idx in levels:
        # for level_idx in range(3,4):
            inv_depth = inv_depth_pyramid[level_idx]
            if inv_depth.dim() == 4:
                inv_depth = inv_depth.squeeze(1)
            # cost_this_level = compute_img_aware_smoothness_cost(inv_depth, self.ref_frame_pyramid[level_idx]/255)/(2**level_idx)
            cost += self.compute_smoothness_cost(inv_depth)/(2**level_idx)
        return cost

    def multi_scale_image_aware_smoothness_cost(self, inv_depth_pyramid, img_pyramid, levels=None, type='lap'):

        # for level_idx in range(self.pyramid_layer_num):
        cost = 0
        if levels is None:
            levels = range(self.pyramid_layer_num)
        for level_idx in levels:
            b, bundle_size, h, w = inv_depth_pyramid[level_idx].shape
            # print(level_idx)
            inv_depth = inv_depth_pyramid[level_idx].view(b*bundle_size, h, w)
            frames = img_pyramid[level_idx].view(b*bundle_size, 3, h, w)
            inv_depth = inv_depth.squeeze(1)
            # cost += compute_img_aware_smoothness_cost(inv_depth, img_pyramid[level_idx])/(2**level_idx)
            if type == 'lap':
                c = self.compute_image_aware_laplacian_smoothness_cost(inv_depth, frames)
            elif type == '1st':
                c = self.compute_image_aware_1st_smoothness_cost(inv_depth, frames)
            elif type == '2nd':
                c = self.compute_image_aware_2nd_smoothness_cost(inv_depth, frames)
            else:
                print("%s not implemented!" %(type))
            cost += (c / (2**level_idx))

        return cost


    def update(self, frames_pyramid, max_itr_num=10):
        frame_num, k, h, w = frames_pyramid[0].size()
        trans_batch = Variable(self.o).expand(frame_num, 3).contiguous()
        trans_batch_prev = Variable(self.o).expand(frame_num, 3).contiguous()

        rot_mat_batch = Variable(self.E).unsqueeze(0).expand(frame_num, 3, 3).contiguous()
        rot_mat_batch_prev = Variable(self.E).unsqueeze(0).expand(frame_num, 3, 3).contiguous()

        pixel_warp = []
        in_view_mask = []

        cur_time = timer()

        for level_idx in range(self.pyramid_layer_num-1, -1, -1):

            max_photometric_cost = self.o.squeeze().expand(frame_num)+10000
            # print(level_idx)
            for i in range(max_itr_num):
                # print(i)
                # cur_time = timer()
                pixel_warp, in_view_mask = self.warp_batch(frames_pyramid[level_idx], level_idx, rot_mat_batch, trans_batch)
                # t_warp += timer()-cur_time

                temporal_grad = pixel_warp - self.ref_frame_pyramid[level_idx].view(3, -1).unsqueeze(0).expand_as(pixel_warp)

                photometric_cost, min_perc_in_view = compute_photometric_cost_norm(temporal_grad.data, in_view_mask.data)
                # print(photometric_cost)
                # print(max_photometric_cost)
                # print(min_perc_in_view)
                # print((photometric_cost < max_photometric_cost).max())
                if min_perc_in_view < .5:
                    break

                if (photometric_cost < max_photometric_cost).max()>0:
                    # print(photometric_cost)
                    trans_batch_prev = trans_batch

                    rot_mat_batch_prev = rot_mat_batch

                    dp = self.LKregress(invH_dIdp=self.invH_dIdp_pyramid[level_idx],
                                        mask=in_view_mask,
                                        It=temporal_grad)
                    st()
                    d_rot_mat_batch = self.twist2mat_batch_func(-dp[:, :, 0:3])
                    trans_batch_new = d_rot_mat_batch.bmm(trans_batch.view(frame_num, 3, 1)).view(frame_num, 3) - dp[:, 3:6]
                    rot_mat_batch_new = d_rot_mat_batch.bmm(rot_mat_batch)

                    trans_list = []
                    rot_list = []
                    for k in range(frame_num):
                        if photometric_cost[k] < max_photometric_cost[k]:
                            rot_list.append(rot_mat_batch_new[k:k+1, :, :])
                            trans_list.append(trans_batch_new[k:k+1, :])
                            max_photometric_cost[k] = photometric_cost[k]
                        else:
                            rot_list.append(rot_mat_batch[k:k+1, :, :])
                            trans_list.append(trans_batch[k:k+1, :])
                    rot_mat_batch = torch.cat(rot_list, 0)
                    trans_batch = torch.cat(trans_list, 0)
                    # if photometric_cost[k] < max_photometric_cost[k]:
                    #     trans_batch = d_rot_mat_batch.bmm(trans_batch.view(frame_num, 3, 1)).view(frame_num, 3) - dp[:, 3:6]
                    #     rot_mat_batch = d_rot_mat_batch.bmm(rot_mat_batch)
                else:
                    break
            rot_mat_batch = rot_mat_batch_prev
            trans_batch = trans_batch_prev

        return rot_mat_batch, trans_batch, frames_pyramid

    def update_with_init_pose(self, frames_pyramid, rot_mat_batch, trans_batch, max_itr_num=10):
        batch_size, frame_num, k, h, w = frames_pyramid[0].size()
        trans_batch_prev = trans_batch
        rot_mat_batch_prev = rot_mat_batch

        pixel_warp = []
        in_view_mask = []

        cur_time = timer()

        for level_idx in range(len(frames_pyramid)-1, -1, -1):
            max_photometric_cost = self.o.expand(batch_size, frame_num)+10000
            # print(level_idx)
            for i in range(max_itr_num):
                # print(i)
                # cur_time = timer()

                pixel_warp, in_view_mask = self.warp_batch(frames_pyramid[level_idx], level_idx, rot_mat_batch, trans_batch)
                # t_warp += timer()-cur_time
                temporal_grad = pixel_warp - self.ref_frame_pyramid[level_idx].view(batch_size, 3, -1).unsqueeze(1).expand_as(pixel_warp)
                photometric_cost, min_perc_in_view = compute_photometric_cost_norm(temporal_grad.data, in_view_mask.data)

                if min_perc_in_view < .5:
                    break

                if (photometric_cost < max_photometric_cost).max()>0:
                    trans_batch_prev = trans_batch
                    rot_mat_batch_prev = rot_mat_batch
                    dp = self.LKregress(invH_dIdp=self.invH_dIdp_pyramid[level_idx],
                                        mask=in_view_mask,
                                        It=temporal_grad)

                    # d_rot_mat_batch = self.twist2mat_batch_func(-dp[:, 0:3])
                    d_rot_mat_batch = self.twist2mat_batch_func(dp[:, :, 0:3]).transpose(2,3)
                    d_rot_mat_batch = d_rot_mat_batch.view(batch_size*frame_num, 3, 3)
                    trans_batch = trans_batch.view(batch_size*frame_num, 3, 1)
                    rot_mat_batch = rot_mat_batch.view(batch_size*frame_num, 3, 3)

                    trans_batch_new = d_rot_mat_batch.bmm(trans_batch).view(batch_size, frame_num, 3) - dp[:, :, 3:6]
                    rot_mat_batch_new = d_rot_mat_batch.bmm(rot_mat_batch)

                    trans_batch_new = trans_batch_new.view(batch_size, frame_num, 3, 1)
                    rot_mat_batch_new = rot_mat_batch_new.view(batch_size, frame_num, 3, 3)
                    trans_batch = trans_batch.view(batch_size, frame_num, 3, 1)
                    rot_mat_batch = rot_mat_batch.view(batch_size, frame_num, 3, 3)

                    trans_list = []
                    rot_list = []
                    # print(photometric_cost)
                    for bk in range(batch_size):
                        for k in range(frame_num):
                            if photometric_cost[bk][k] < max_photometric_cost[bk][k]:
                                rot_list.append(rot_mat_batch_new[bk:bk+1, k:k+1, :, :])
                                trans_list.append(trans_batch_new[bk:bk+1, k:k+1, :])
                                max_photometric_cost[bk][k] = photometric_cost[bk][k]
                            else:
                                rot_list.append(rot_mat_batch[bk:bk+1, k:k+1, :, :])
                                trans_list.append(trans_batch[bk:bk+1, k:k+1, :])
                    rot_mat_batch = torch.cat(rot_list, 0).view(batch_size, frame_num, 3, 3)
                    trans_batch = torch.cat(trans_list, 0).view(batch_size, frame_num, 3)
                else:
                    break
            rot_mat_batch = rot_mat_batch_prev
            trans_batch = trans_batch_prev
        return rot_mat_batch, trans_batch


    def forward(self, ref_frame_pyramid, src_frame_pyramid, ref_inv_depth_pyramid, max_itr_num=10):
            self.init(ref_frame_pyramid=ref_frame_pyramid, inv_depth_pyramid=ref_inv_depth_pyramid)
            rot_mat_batch, trans_batch, src_frames_pyramid = self.update(src_frame_pyramid, max_itr_num=max_itr_num)
            return rot_mat_batch, trans_batch
