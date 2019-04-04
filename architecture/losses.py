from __future__ import absolute_import
import sys

import torch
from torch import nn

EPSILON = 1e-10

class KL_divergence(nn.Module):
    """
        Computes KL-divergence loss for inputs that are logits
    """
    def __init__(self, reduction = 'elementwise_mean'):
        super(KL_divergence, self).__init__()
        self.loss_fn = nn.MSELoss()
        if reduction == 'elementwise_mean':
            self.reduction = torch.mean
        elif reduction == 'elementwise_sum':
            self.reduction = torch.sum
        else:
            raise Exception("invalid reduction for logits KL divergence")

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction tensors of logits
            targets: ground truth tensor of logits
        """
        #loss_elementwise = torch.exp(inputs)*(inputs - targets)
        #loss_elementwise = (torch.abs(inputs - targets))
        #loss_elementwise = torch.exp(inputs)*((inputs - targets) * (inputs - targets))
        #loss_elementwise = (inputs - targets) * (inputs - targets)
        #loss = -self.reduction(loss_elementwise)
        #loss = torch.mean(torch.abs(torch.exp(inputs) - torch.exp(targets)))
        #loss = self.loss_fn(torch.exp(inputs), torch.exp(targets))
        
        n_inputs = inputs + EPSILON
        #js = n_inputs*(torch.log(n_inputs) - torch.log(targets)) + targets*(torch.log(targets) - torch.log(n_inputs))
        #js /= 2.
        #loss = self.reduction(js)

        js = torch.exp(-inputs)*(-inputs - torch.log(targets)) + targets*(torch.log(targets) + inputs)
        js /= 2.
        loss = -self.reduction(js)

        return loss

class quad_mse_1(nn.Module):
    def __init__(self, reduction = 'elementwise_mean'):
        super(quad_mse_1, self).__init__()
        self.loss_fn = nn.MSELoss()
        
    def forward(self, density_maps, gt_image):
        loss = 0
        gts_level  = [gt_image]
        for level in density_maps:
            for ind, pred in enumerate(level):
                #print(ind, "-->", pred.size(), gts_level[ind].size())
                loss += self.loss_fn(pred, gts_level[ind])
            new_gts = []
            for gt in gts_level:
                _, _, w, h = gt.size()

                hh = h if h % 2 == 0 else h - 1
                ww = w if w % 2 == 0 else w - 1

                gt_1 =  gt[:, :, 0:ww//2, 0:hh//2]
                gt_2 =  gt[:, :, 0:ww//2, hh//2:hh]
                gt_3 =  gt[:, :, ww//2:ww, 0:hh//2]
                gt_4 =  gt[:, :, ww//2:ww, hh//2:hh]
                new_gts.append(gt_1)
                new_gts.append(gt_2)
                new_gts.append(gt_3)
                new_gts.append(gt_4)
            gts_level = new_gts

        return loss


class quad_mse_2_1(nn.Module):
    def __init__(self, reduction = 'elementwise_mean', discriminator_threshold = 20):
        super(quad_mse_2_1, self).__init__()
        self.loss_mse = nn.MSELoss()
        self.loss_xent = nn.BCELoss()
        self.discriminator_threshold = discriminator_threshold
    
    def forward(self, density_maps, discriminators, decoders, gt_image):
        loss_mse = 0
        loss_xent = 0
        gts_level  = [gt_image]
        loss_mse_decoder = 0
        assert len(density_maps) == len(discriminators) and discriminators[-1] == [], "error constructing quadtree {} {}".format(len(discriminators), len(density_maps))
        for l, (level_den, level_dec) in enumerate(zip(density_maps, decoders)):
            for ind, (pred_den, pred_dec) in enumerate(zip(level_den, level_dec)):
                loss_mse_decoder += self.loss_mse(pred_dec, gts_level[ind])
                if l + 1 < len(discriminators):
                    people_cnt = torch.sum(gts_level[ind].contiguous().view(gts_level[ind].size(0), -1), dim = 1)
                    gt_xent = torch.ones(people_cnt.size())
                    gt_xent[people_cnt < self.discriminator_threshold] = 0
                    gt_xent = gt_xent.view(gt_xent.size(0), 1) #add one dimension 
                    assert gt_xent.size() == discriminators[l][ind].size(),  "error in xent shape"
                    gt_xent = gt_xent.cuda()
                    loss_xent += self.loss_xent(discriminators[l][ind], gt_xent)

            new_gts = []
            for gt in gts_level:
                chunks = torch.chunk(gt, chunks = 2, dim = 2)
                gt_1, gt_2 = torch.chunk(chunks[0], chunks = 2, dim = 3)
                gt_3, gt_4 = torch.chunk(chunks[1], chunks = 2, dim = 3)

                new_gts.append(gt_1)
                new_gts.append(gt_2)
                new_gts.append(gt_3)
                new_gts.append(gt_4)
            gts_level = new_gts

        loss_mse = self.loss_mse(density_maps[0][0], gt_image) + loss_mse_decoder
        loss = loss_xent + loss_mse
        return loss

class quad_mse_2_2(nn.Module):
    def __init__(self, reduction = 'elementwise_mean', discriminator_threshold = 20):
        super(quad_mse_2_2, self).__init__()
        self.loss_mse = nn.MSELoss()
        self.loss_xent = nn.BCELoss()
        self.discriminator_threshold = discriminator_threshold
    
    def forward(self, density_maps, discriminators, decoders, gt_image):
        loss_mse_fusion = 0
        loss_mse_decoder = 0
        loss_xent = 0
        gts_level  = [gt_image]
        assert len(density_maps) == len(discriminators) and discriminators[-1] == [], "error constructing quadtree {} {}".format(len(discriminators), len(density_maps))
        for l, (level_den, level_dec) in enumerate(zip(density_maps, decoders)):
            for ind, (pred_den, pred_dec) in enumerate(zip(level_den, level_dec)):
                #print(ind, "-->", pred.size(), gts_level[ind].size())
                loss_mse_fusion += self.loss_mse(pred_den, gts_level[ind])
                loss_mse_decoder += self.loss_mse(pred_dec, gts_level[ind])
                if l + 1 < len(discriminators):
                    people_cnt = torch.sum(gts_level[ind].contiguous().view(gts_level[ind].size(0), -1, 1), dim = 1)
                    gt_xent = torch.ones(people_cnt.size())
                    gt_xent[people_cnt < self.discriminator_threshold] = 0
                    gt_xent = gt_xent.view(gt_xent.size(0), 1) #add one dimension 
                    assert gt_xent.size() == discriminators[l][ind].size(),  "error in xent shape {} {}".format(gt_xent.size(), discriminators[l][ind].size())
                    gt_xent = gt_xent.cuda()
                    loss_xent += self.loss_xent(discriminators[l][ind], gt_xent)

            new_gts = []
            for gt in gts_level:
                chunks = torch.chunk(gt, chunks = 2, dim = 2)
                gt_1, gt_2 = torch.chunk(chunks[0], chunks = 2, dim = 3)
                gt_3, gt_4 = torch.chunk(chunks[1], chunks = 2, dim = 3)

                new_gts.append(gt_1)
                new_gts.append(gt_2)
                new_gts.append(gt_3)
                new_gts.append(gt_4)
            gts_level = new_gts

        loss = loss_xent + loss_mse_fusion + loss_mse_decoder
        return loss

class quad_mse_2_2_2(nn.Module):
    """
    Implements loss between quadnet levels using MSE for density maps + MSE for people count 
    """
    def __init__(self, reduction = 'elementwise_mean', discriminator_threshold = 20):
        super(quad_mse_2_2_2, self).__init__()
        self.loss_mse = nn.MSELoss()
        self.loss_xent = nn.BCELoss()
        self.loss_cnt = nn.MSELoss()
        self.discriminator_threshold = discriminator_threshold
    
    def forward(self, density_maps, discriminators, decoders, gt_image):
        loss_mse_fusion = 0
        loss_mse_decoder = 0
        loss_mse_cnt_fusion = 0
        loss_mse_cnt_decoder = 0
        loss_xent = 0
        gts_level  = [gt_image]
        assert len(density_maps) == len(discriminators) and discriminators[-1] == [], "error constructing quadtree {} {}".format(len(discriminators), len(density_maps))
        batch_size = gt_image.size()[0]
        for l, (level_den, level_dec) in enumerate(zip(density_maps, decoders)):
            for ind, (pred_den, pred_dec) in enumerate(zip(level_den, level_dec)):
                #print(ind, "-->", pred.size(), gts_level[ind].size())
                loss_mse_fusion += self.loss_mse(pred_den, gts_level[ind])
                loss_mse_decoder += self.loss_mse(pred_dec, gts_level[ind])
                loss_mse_cnt_fusion += self.loss_cnt(torch.sum(pred_den.contiguous().view(batch_size, -1), dim = 1), torch.sum(gts_level[ind].contiguous().view(batch_size, -1), dim = 1))
                loss_mse_cnt_decoder += self.loss_cnt(torch.sum(pred_dec.contiguous().view(batch_size, -1), dim = 1), torch.sum(gts_level[ind].contiguous().view(batch_size, -1), dim = 1))
                if l + 1 < len(discriminators):
                    people_cnt = torch.sum(gts_level[ind].contiguous().view(gts_level[ind].size(0), -1, 1), dim = 1)
                    gt_xent = torch.ones(people_cnt.size())
                    gt_xent[people_cnt < self.discriminator_threshold] = 0
                    gt_xent = gt_xent.view(gt_xent.size(0), 1) #add one dimension 
                    assert gt_xent.size() == discriminators[l][ind].size(),  "error in xent shape {} {}".format(gt_xent.size(), discriminators[l][ind].size())
                    gt_xent = gt_xent.cuda()
                    loss_xent += self.loss_xent(discriminators[l][ind], gt_xent)

            new_gts = []
            for gt in gts_level:
                chunks = torch.chunk(gt, chunks = 2, dim = 2)
                gt_1, gt_2 = torch.chunk(chunks[0], chunks = 2, dim = 3)
                gt_3, gt_4 = torch.chunk(chunks[1], chunks = 2, dim = 3)

                new_gts.append(gt_1)
                new_gts.append(gt_2)
                new_gts.append(gt_3)
                new_gts.append(gt_4)
            gts_level = new_gts

        loss = loss_xent + loss_mse_fusion + loss_mse_decoder + loss_mse_cnt_fusion + loss_mse_cnt_decoder
        return loss

class quad_mse_2_3(nn.Module):
    def __init__(self, reduction = 'elementwise_mean', discriminator_threshold = 20):
        super(quad_mse_2_3, self).__init__()
        self.loss_mse = nn.MSELoss()
        self.loss_xent = nn.BCELoss()
        self.discriminator_threshold = discriminator_threshold
    
    def forward(self, density_maps, discriminators, decoders, intermediates, gt_image, gt_image_small):
        loss_mse_fusion = 0
        loss_mse_decoder = 0
        loss_mse_mid = 0
        loss_xent = 0
        gts_level  = [gt_image]
        gts_level_small  = [gt_image_small]
        assert len(density_maps) == len(discriminators) and discriminators[-1] == [], "error constructing quadtree {} {}".format(len(discriminators), len(density_maps))
        for l, (level_den, level_dec, level_inter) in enumerate(zip(density_maps, decoders, intermediates)):
            for ind, (pred_den, pred_dec, pred_inter) in enumerate(zip(level_den, level_dec, level_inter)):
                #print(ind, "-->", pred.size(), gts_level[ind].size())
                loss_mse_fusion += self.loss_mse(pred_den, gts_level[ind])
                loss_mse_decoder += self.loss_mse(pred_dec, gts_level[ind])
                loss_mse_mid += self.loss_mse(pred_inter, gts_level_small[ind])
                if l + 1 < len(discriminators):
                    people_cnt = torch.sum(gts_level[ind].contiguous().view(gts_level[ind].size(0), -1, 1), dim = 1)
                    gt_xent = torch.ones(people_cnt.size())
                    gt_xent[people_cnt < self.discriminator_threshold] = 0
                    gt_xent = gt_xent.view(gt_xent.size(0), 1) #add one dimension 
                    assert gt_xent.size() == discriminators[l][ind].size(),  "error in xent shape {} {}".format(gt_xent.size(), discriminators[l][ind].size())
                    gt_xent = gt_xent.cuda()
                    loss_xent += self.loss_xent(discriminators[l][ind], gt_xent)

            new_gts = []
            for gt in gts_level:
                chunks = torch.chunk(gt, chunks = 2, dim = 2)
                gt_1, gt_2 = torch.chunk(chunks[0], chunks = 2, dim = 3)
                gt_3, gt_4 = torch.chunk(chunks[1], chunks = 2, dim = 3)

                new_gts.append(gt_1)
                new_gts.append(gt_2)
                new_gts.append(gt_3)
                new_gts.append(gt_4)
            gts_level = new_gts

            new_gts_small = []
            for gt in gts_level_small:
                chunks = torch.chunk(gt, chunks = 2, dim = 2)
                gt_1, gt_2 = torch.chunk(chunks[0], chunks = 2, dim = 3)
                gt_3, gt_4 = torch.chunk(chunks[1], chunks = 2, dim = 3)

                new_gts_small.append(gt_1)
                new_gts_small.append(gt_2)
                new_gts_small.append(gt_3)
                new_gts_small.append(gt_4)
            gts_level_small = new_gts_small

        loss = loss_xent + loss_mse_fusion + loss_mse_decoder + loss_mse_mid
        return loss

class quad_mse_2_2_steps(nn.Module):
    def __init__(self, reduction = 'elementwise_mean', discriminator_threshold = 20, fusion_steps = 2):
        super(quad_mse_2_2_steps, self).__init__()
        self.fusion_steps = fusion_steps
        self.loss_mse = nn.MSELoss()
        self.loss_xent = nn.BCELoss()
        self.discriminator_threshold = discriminator_threshold
    
    def forward(self, density_maps, discriminators, decoders, gt_image):
        loss_mse_fusion = 0
        loss_mse_decoder = 0
        loss_xent = 0
        gts_level  = [gt_image]
        assert len(density_maps) == len(discriminators) and discriminators[-1] == [], "error constructing quadtree {} {}".format(len(discriminators), len(density_maps))
        for l, (level_den, level_dec) in enumerate(zip(density_maps, decoders)):
            for ind, (pred_den, pred_dec) in enumerate(zip(level_den, level_dec)):
                #print(ind, "-->", pred.size(), gts_level[ind].size())
                for i in range(self.fusion_steps):
                    loss_mse_fusion += self.loss_mse(pred_den[i], gts_level[ind])
                loss_mse_decoder += self.loss_mse(pred_dec, gts_level[ind])
                if l + 1 < len(discriminators):
                    people_cnt = torch.sum(gts_level[ind].contiguous().view(gts_level[ind].size(0), -1, 1), dim = 1)
                    gt_xent = torch.ones(people_cnt.size())
                    gt_xent[people_cnt < self.discriminator_threshold] = 0
                    gt_xent = gt_xent.view(gt_xent.size(0), 1) #add one dimension 
                    assert gt_xent.size() == discriminators[l][ind].size(),  "error in xent shape {} {}".format(gt_xent.size(), discriminators[l][ind].size())
                    gt_xent = gt_xent.cuda()
                    loss_xent += self.loss_xent(discriminators[l][ind], gt_xent)

            new_gts = []
            for gt in gts_level:
                chunks = torch.chunk(gt, chunks = 2, dim = 2)
                gt_1, gt_2 = torch.chunk(chunks[0], chunks = 2, dim = 3)
                gt_3, gt_4 = torch.chunk(chunks[1], chunks = 2, dim = 3)

                new_gts.append(gt_1)
                new_gts.append(gt_2)
                new_gts.append(gt_3)
                new_gts.append(gt_4)
            gts_level = new_gts

        loss = loss_xent + loss_mse_fusion + loss_mse_decoder
        return loss