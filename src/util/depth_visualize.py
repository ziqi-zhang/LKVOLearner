import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as st

def gray2rgb(im, cmap='plasma'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 3)
    return np.squeeze(rgb_img)

def vis_depthmap(input):
    x = (input-input.min()) * (1/(input.max()-input.min()+.00001))
    x = np.expand_dims(x, axis=2)
    # x = np.repeat(x, 3, axis=2)
    return gray2rgb(x)
