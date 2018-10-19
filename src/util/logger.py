import os, sys
from tensorboardX import SummaryWriter

class Logger:
    def __init__(self, output_dir):
        self.logger = SummaryWriter(os.path.join(output_dir))

    def add_scalar(self, name, value, iter):
        self.logger.add_scalar(name, value, iter)
