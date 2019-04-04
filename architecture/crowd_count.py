import torch.nn as nn
import architecture.network as network
from architecture.models import MCNN, MCNN_1, SIMPLE, CROWD_UNET, QUADTREE_1, QUADTREE_2_1, QUADTREE_2_2, QUADTREE_2_3, QUADTREE_2_4, QUADTREE_3_1, QUADTREE_MCNN, QUADTREE_STEPS
from architecture.losses import KL_divergence, quad_mse_1, quad_mse_2_1, quad_mse_2_2, quad_mse_2_2_2, quad_mse_2_3, quad_mse_2_2_steps


class CrowdCounter(nn.Module):
    def __init__(self, loss = 'MSE', model = 'MCNN', loss_quad = 'all', fusion_layer = 1, fusion_steps = 2):
        super(CrowdCounter, self).__init__()        
        self.loss_quad = loss_quad
        self.fusion_steps = fusion_steps
        if loss == 'MSE':
            self.loss_fn = nn.MSELoss()
        elif loss == 'KL-divergence':
            self.loss_fn = KL_divergence()
        else:
            raise Exception("invalid loss function for network")

        self.model = model
        if model == 'MCNN':
            self.DME = MCNN()
        elif model == 'MCNN_1':
            self.DME = MCNN_1()
        elif model == 'SIMPLE':
            self.DME = SIMPLE()
        elif model == 'UNET':
            self.DME = CROWD_UNET()
        elif model == 'AUTOENCODER':
            self.DME = QUADTREE_2_4()
        elif model == 'QUAD':
            self.DME = QUADTREE_2_2(fusion = fusion_layer) #shared weights between levels
            if loss_quad == 'top':
                self.loss_fn = quad_mse_2_1()
                print("using top loss")
            elif loss_quad == 'all':
                self.loss_fn = quad_mse_2_2()
                print("using loss in each level")
            elif loss_quad == 'cnt':
                self.loss_fn = quad_mse_2_2_2()
                print("using loss in each level + cnt loss for people")
            elif loss_quad == 'mix':
                print("using loss mixed between 'cnt' and 'all' losses")
            else:
                raise Exception("invalid quad loss '{}', expected ['top', 'all', 'cnt']".format(loss_quad))
        elif model == 'QUAD-MCNN':
            self.DME = QUADTREE_MCNN() #shared weights between levels
            if loss_quad == 'top':
                self.loss_fn = quad_mse_2_1()
                print("using top loss")            
            elif loss_quad == 'all':
                self.loss_fn = quad_mse_2_2()
                print("using loss in each level")
            elif loss_quad == 'cnt':
                self.loss_fn = quad_mse_2_2_2()
                print("using loss in each level + cnt loss for people")
            else:
                raise Exception("invalid quad loss '{}', expected ['top', 'all', 'cnt']".format(loss_quad))
        elif model == 'QUAD-MID':
            self.DME = QUADTREE_2_3() #shared weights between levels
            self.loss_fn = quad_mse_2_3()
        elif model == 'QUAD-STEPS':
            self.DME = QUADTREE_STEPS(fusion = fusion_layer, fusion_steps = fusion_steps) # using multiple fusion steps
            self.loss_fn = quad_mse_2_2_steps(fusion_steps = fusion_steps)
        else:
            raise Exception("invalid network model")

    def forward(self,  im_data, gt_data=None, gt_data_small = None, epoch = 0):
        import time
        im_data = network.np_to_variable(im_data, is_cuda=True, is_training=self.training)
        start = time.time()
        if not self.model.startswith('QUAD'):
            density_map = self.DME(im_data)
        else:
            if self.model == 'QUAD-STEPS':
                reconstruction_steps = self.DME(im_data, level = 0)
                density_map = reconstruction_steps[-1]
            else:
                density_map = self.DME(im_data, level = 0)
        end = time.time()
        elapsed = end - start
        #print ("forward {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))

        if self.training:
            start = time.time()
            gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=self.training)
            if not self.model.startswith('QUAD'):
                self.loss = self.loss_fn(density_map, gt_data)
            else:
                if self.model == 'QUAD-MID':
                    gt_data_small = network.np_to_variable(gt_data_small, is_cuda=True, is_training=self.training)
                    self.loss = self.loss_fn(self.DME.reconstructions, self.DME.discriminators, self.DME.upsamples, self.DME.intermediates, gt_data, gt_data_small)
                else:
                    if self.loss_quad == 'mix':
                        if epoch % 4 == 3:#use 'cnt' loss every 4 epochs
                            self.loss_fn = quad_mse_2_2_2()
                        else:
                            self.loss_fn = quad_mse_2_2()

                    self.loss = self.loss_fn(self.DME.reconstructions, self.DME.discriminators, self.DME.upsamples, gt_data)
            end = time.time()
            elapsed = end - start
            #print ("loss {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
        return density_map
