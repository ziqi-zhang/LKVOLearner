import torch
from torch import FloatTensor
from torch.autograd import Variable
from timeit import default_timer as timer
from torch.nn.functional import grid_sample
from torch.nn import ReplicationPad2d

from pdb import set_trace as st

def grid_bilinear_sampling(A, x, y):
    b, ref_num, k, h, w = A.size()
    x_norm = x/((w-1)/2) - 1
    y_norm = y/((h-1)/2) - 1
    grid = torch.cat((x_norm.view(b, ref_num, h, w, 1), \
                      y_norm.view(b, ref_num, h, w, 1)), 4)
    A = A.view(b*ref_num, k, h, w)
    grid = grid.view(b*ref_num, h, w, 2)
    Q = grid_sample(A, grid, mode='bilinear')
    in_view_mask = Variable(((x_norm.data > -1+2/w) & (x_norm.data < 1-2/w) \
                              & (y_norm.data > -1+2/h) & (y_norm.data < 1-2/h))\
                            .type_as(A.data))
    # in_view_mask = Variable(((x.data > 1) & (x.data < w-2) & (y.data > 1) & (y.data < h-2)).type_as(A.data))
    # in_view_mask = Variable(((x.data > -1+3/w) & (x.data < 1-3/w) & (y.data > -1+3/h) & (y.data < 1-3/h)).type_as(A.data))
    Q = Q.view(b, ref_num, k, h*w)
    return Q, in_view_mask
