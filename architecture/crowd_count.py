import torch.nn as nn
import architecture.network as network
from architecture.models import MCNN


class CrowdCounter(nn.Module):
    def __init__(self):
        super(CrowdCounter, self).__init__()
        self.loss_fn = nn.MSELoss()
        self.net = MCNN()

    def forward(self,  im_data, gt_data=None):
        im_data = network.np_to_variable(im_data, is_cuda=True, is_training=self.training)
        density_map = self.net(im_data)

        if self.training:
            gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=self.training)
            self.loss = self.loss_fn(density_map, gt_data)
        return density_map
