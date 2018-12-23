import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as st

def gray2rgb(im, cmap='plasma'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, -1)
    return np.squeeze(rgb_img)

def vis_depthmap(input):
    x = (input-input.min()) * (1/(input.max()-input.min()+.00001))
    x = np.expand_dims(x, axis=2)
    # x = np.repeat(x, 3, axis=2)
    return gray2rgb(x)

def img_norm2one(input):
    x = (input-input.min()) * (1/(input.max()-input.min()+.00001))
    return x

def vis_warp_error(img, warped_img):
    mask = warped_img.sum(2)>0
    warp_error = (img-warped_img)
    warp_error = warp_error.sum(2) * mask
    warp_error = img_norm2one(warp_error)
    warp_error = gray2rgb(warp_error)
    return warp_error*255

def image_range_to_255(input):
    x = (input-input.min()) * (255/(input.max()-input.min()+.00001))
    return x

def gray2attention_map(im, cmap='seismic'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 3)
    return np.squeeze(rgb_img)

def tensor2numpy(tensor):
    assert type(tensor)==torch.Tensor
    if tensor.requires_grad==True:
        return tensor.cpu().detach().numpy()
    else:
        return tensor.cpu().numpy()

def feat_map2rgb(feat):
    im = tensor2numpy(feat)
    im = gray2attention_map(im)
    im = image_range_to_255(im)
    return im
