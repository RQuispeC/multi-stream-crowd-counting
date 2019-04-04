import torch
import torch.nn as nn
from torch.nn import functional as F
from architecture.network import Conv2d
import torchvision
import numpy as np

class MCNN(nn.Module):
    '''
    Multi-column CNN 
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''
    
    def __init__(self, bn=False):
        super(MCNN, self).__init__()
        
        self.branch0 = nn.Sequential(Conv2d( 1, 12, 11, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(12, 24, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 12, 9, same_padding=True, bn=bn),
                                     Conv2d(12,  6, 9, same_padding=True, bn=bn))

        self.branch1 = nn.Sequential(Conv2d( 1, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(32, 16, 7, same_padding=True, bn=bn),
                                     Conv2d(16,  8, 7, same_padding=True, bn=bn))
        
        self.branch2 = nn.Sequential(Conv2d( 1, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5, same_padding=True, bn=bn),
                                     Conv2d(20, 10, 5, same_padding=True, bn=bn))
        
        self.branch3 = nn.Sequential(Conv2d( 1, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                     Conv2d(24, 12, 3, same_padding=True, bn=bn))
        
        self.fuse = nn.Sequential(Conv2d( 36, 1, 1, same_padding=True, bn=bn))
        
    def forward(self, im_data):
        x0 = self.branch0(im_data)
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x0,x1,x2,x3),1)
        x = self.fuse(x)
        
        return x


class MCNN_1(nn.Module):
    '''
    Multi-column CNN_1 creating intermediate conection between streams
    '''
    
    def __init__(self, bn=False):
        super(MCNN_1, self).__init__()
        
        self.branch0_0 = nn.Sequential(Conv2d( 1, 12, 11, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(12, 24, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2))
        self.branch0_1 = nn.Sequential(Conv2d(56, 28, 9, same_padding=True, bn=bn),
                                     Conv2d(28, 14, 9, same_padding=True, bn=bn))

        self.branch1_0 = nn.Sequential(Conv2d( 1, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2))
        self.branch1_1 = nn.Sequential(Conv2d(72, 36, 7, same_padding=True, bn=bn),
                                     Conv2d(36, 18, 7, same_padding=True, bn=bn))
        
        self.branch2_0 = nn.Sequential(Conv2d( 1, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2))
        self.branch2_1 = nn.Sequential(Conv2d(88, 44, 5, same_padding=True, bn=bn),
                                     Conv2d(44, 22, 5, same_padding=True, bn=bn))
        
        self.branch3_0 = nn.Sequential(Conv2d( 1, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2))
        self.branch3_1 = nn.Sequential(Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                     Conv2d(24, 12, 3, same_padding=True, bn=bn))
        
        self.fuse_1 = nn.Sequential(Conv2d(66, 33, 1, same_padding=True, bn=bn))
        self.fuse_2 = nn.Sequential(Conv2d(33, 1, 1, same_padding=True, bn=bn))
        
    def forward(self, im_data):
        x0 = self.branch0_0(im_data)
        x1 = self.branch1_0(im_data)
        x2 = self.branch2_0(im_data)
        x3 = self.branch3_0(im_data)

        x_0_1 = torch.cat((x0, x1), 1)
        x_1_2 = torch.cat((x1, x2), 1)
        x_2_3 = torch.cat((x2, x3), 1)

        nx0 = self.branch0_1(x_0_1)
        nx1 = self.branch1_1(x_1_2)
        nx2 = self.branch2_1(x_2_3)
        nx3 = self.branch3_1(x3)

        x = torch.cat((nx0,nx1,nx2,nx3),1)
        x = self.fuse_1(x)
        x = self.fuse_2(x)
        
        return x

class SIMPLE(nn.Module):
    '''
    SIMPLE model for gradient test
    '''
    
    def __init__(self, bn=False):
        super(SIMPLE, self).__init__()
        
        self.s = nn.Sequential(Conv2d( 1, 2, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     #nn.Conv2d(2, 1, 3, 1, padding=1),
                                     Conv2d( 2, 1, 3, same_padding=True, bn=bn),
                                     #nn.LeakyReLU(inplace=True),
                                     nn.MaxPool2d(2))
        
    def forward(self, im_data):
        x = self.s(im_data)
        
        return x

################################### QUADTREE modules ###################################33
class encoder_resnet50(nn.Module):

    def __init__(self):
        super(encoder_resnet50, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers0 = Conv2d( 1, 3, 1, same_padding=True, bn=False) #convert gray image to a 3 channel tensor
        self.layers1 = nn.Sequential(base[0], base[1], base[2]) # 64 channels
        self.layers2 = nn.Sequential(base[3], base[4]) # 256 channels
        self.layers3 = base[5] # scaled by 8 and 512 channels


    def forward(self, x):
        '''
        x: batch of input images
        '''
        x0 = self.layers0(x)
        x1 = self.layers1(x0)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        return x0, x1, x2, x3

class decoder_resnet50(nn.Module):

    def __init__(self):
        super(decoder_resnet50, self).__init__()
        self.upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = Conv2d(512, 256, 3, same_padding=True, bn=False)
        self.upsample2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.conv2 = Conv2d(128, 64, 3, same_padding=True, bn=False)
        self.upsample3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.density = Conv2d(35, 1, 1, same_padding=True, bn=False)

    def forward(self, x0, x1, x2, x3):
        '''
        x0, x1, x2, x3: zero, first, second and third output of encoders
        '''
        x3_up = self.upsample1(x3)
        #update x3_up to same size of x2 (for odd sizes of tensors)
        x3_up = F.upsample(x3_up, size = (x2.size()[-2], x2.size()[-1]), mode = 'bilinear')
        x3_up_x2 = torch.cat((x3_up, x2), dim = 1)
        x2_up = self.conv1(x3_up_x2)

        x2_up = self.upsample2(x2_up)
        #update x2_up to same size of x1 (for odd sizes of tensors)
        x2_up = F.upsample(x2_up, size = (x1.size()[-2], x1.size()[-1]), mode = 'bilinear')
        x2_up_x1 = torch.cat((x2_up, x1), dim = 1)
        x1_up = self.conv2(x2_up_x1)


        x1_up = self.upsample3(x1_up)
        #update x2_up to same size of x1 (for odd sizes of tensors)
        x1_up = F.upsample(x1_up, size = (x0.size()[-2], x0.size()[-1]), mode = 'bilinear')
        x1_up_x0 = torch.cat((x1_up, x0), dim = 1)
        density_map = self.density(x1_up_x0)

        return density_map


class MCNN_encoder(nn.Module):
    '''
    Encoder based on Multi-column CNN 
    '''
    
    def __init__(self, bn=False):
        super(MCNN_encoder, self).__init__()
        
        self.branch0 = nn.Sequential(Conv2d( 1, 12, 11, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(12, 24, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 12, 9, same_padding=True, bn=bn))

        self.branch1 = nn.Sequential(Conv2d( 1, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(32, 16, 7, same_padding=True, bn=bn))
        
        self.branch2 = nn.Sequential(Conv2d( 1, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5, same_padding=True, bn=bn))
        
        self.branch3 = nn.Sequential(Conv2d( 1, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(48, 24, 3, same_padding=True, bn=bn))
        self.fusion = Conv2d(72, 128, 1, same_padding=True, bn=False)

    def forward(self, im_data):
        x0 = self.branch0(im_data)
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x0,x1,x2,x3), 1) #channels size 72

        x = self.fusion(x)
        return x

class MCNN_decoder(nn.Module):
    def __init__(self, bn = False):
        super(MCNN_decoder, self).__init__()
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2)
        self.conv1 = Conv2d(64, 32, 3, same_padding=True, bn=False)        
        self.upsample2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2)
        self.density = Conv2d(16, 1, 1, same_padding=True, bn=False)

    def forward(self, im_data, decoded):
        x = self.upsample1(decoded)
        x = self.conv1(x)
        x = self.upsample2(x)
        x = F.upsample(x, size = (im_data.size()[-2], im_data.size()[-1]), mode = 'bilinear')
        den = self.density(x)
        return den

class encoder(nn.Module):
    def __init__(self, bn = False):
        super(encoder, self).__init__()
        self.downsample1 = nn.Sequential(Conv2d( 1, 12, 11, same_padding=True, bn=bn), 
                                        nn.MaxPool2d(2))
        self.downsample2 = nn.Sequential(Conv2d(12, 24, 9, same_padding=True, bn=bn),
                                        nn.MaxPool2d(2))
        self.embed = nn.Sequential(Conv2d(24, 48, 9, same_padding=True, bn=bn),
                                     Conv2d(48, 96, 9, same_padding=True, bn=bn))
    def forward(self, x):
        x1 = self.downsample1(x)
        x2 = self.downsample2(x1)
        x3 = self.embed(x2)
        return x, x1, x2, x3

class decoder(nn.Module):
    def __init__(self, bn = False):
        super(decoder, self).__init__()
        self.upsample1 = nn.ConvTranspose2d(96, 48, kernel_size=3, stride=1)
        self.conv1 = Conv2d(72, 36, 3, same_padding=True, bn=False)
        self.upsample2 = nn.ConvTranspose2d(36, 18, kernel_size=3, stride=2)
        self.conv2 = Conv2d(30, 15, 3, same_padding=True, bn=False)
        self.upsample3 = nn.ConvTranspose2d(15, 8, kernel_size=3, stride=2)
        self.density = Conv2d(8, 1, 1, same_padding=True, bn=False)

    def forward(self, x, x1, x2, x3):
        x3_up = self.upsample1(x3)
        x3_up = F.upsample(x3_up, size = (x2.size()[-2], x2.size()[-1]), mode = 'bilinear')
        x3_up_x2 = torch.cat((x3_up, x2), dim = 1)
        x2_up = self.conv1(x3_up_x2)

        x2_up = self.upsample2(x2_up)
        x2_up = F.upsample(x2_up, size = (x1.size()[-2], x1.size()[-1]), mode = 'bilinear')
        x2_up_x1 = torch.cat((x2_up, x1), dim = 1)
        x1_up = self.conv2(x2_up_x1)


        den = self.upsample3(x1_up)
        den = F.upsample(den, size = (x.size()[-2], x.size()[-1]), mode = 'bilinear')
        den = self.density(den)
        return den

class ListModule(object):
    #Should work with all kind of module
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))

class CROWD_UNET(nn.Module):
    '''
    UNet adaptation for crowd counting
    '''
    
    def __init__(self, levels = 4):
        super(CROWD_UNET, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder()

    def forward(self, im_data):

        x, x1, x2, x3 = self.encoder(im_data)
        density_map = self.decoder(x, x1, x2, x3)
        density_map = F.upsample(density_map, size = (im_data.size()[-2], im_data.size()[-1]), mode = 'bilinear')

        return density_map


class fusion_1(nn.Module):
    def __init__(self, bn = False):
        print("USING FUSION 1")
        super(fusion_1, self).__init__()
        self.fusion_layer = nn.Sequential(Conv2d(3, 64, 1, same_padding=True, bn=False),
                                    Conv2d(64, 32, 9, same_padding=True, bn=False),
                                    Conv2d(32, 16, 7, same_padding=True, bn=False),
                                    Conv2d(16, 8, 5, same_padding=True, bn=False),
                                    Conv2d(8, 1, 1, same_padding=True, bn=False))

    def forward(self, current_level, next_level, image = "", alpha_image = 1./255.):
        features = torch.cat((current_level, next_level, image*alpha_image), dim = 1)
        #features = torch.cat((current_level, next_level), dim = 1)
        den = self.fusion_layer(features)
        return den

class fusion_2(nn.Module):
    """
    simplified version of fusion 1
    """
    def __init__(self, bn = False):
        print("USING FUSION 2")
        super(fusion_2, self).__init__()
        self.fusion_layer = nn.Sequential(Conv2d(3, 8, 3, same_padding=True, bn=False),
                                    Conv2d(8, 16, 3, same_padding=True, bn=False),
                                    Conv2d(16, 8, 3, same_padding=True, bn=False),
                                    Conv2d(8, 1, 1, same_padding=True, bn=False))

    def forward(self, current_level, next_level, image = "", alpha_image = 1./255.):
        features = torch.cat((current_level, next_level, image*alpha_image), dim = 1)
        den = self.fusion_layer(features)
        return den


class fusion_3(nn.Module):
    """
    fusion 2 + removing image as input
    """
    def __init__(self, bn = False):
        print("USING FUSION 3")
        super(fusion_3, self).__init__()
        self.fusion_layer = nn.Sequential(Conv2d(2, 8, 3, same_padding=True, bn=False),
                                    Conv2d(8, 16, 3, same_padding=True, bn=False),
                                    Conv2d(16, 8, 3, same_padding=True, bn=False),
                                    Conv2d(8, 1, 1, same_padding=True, bn=False))

    def forward(self, current_level, next_level, image = "", alpha_image = 1./255.):
        features = torch.cat((current_level, next_level), dim = 1)
        den = self.fusion_layer(features)
        return den

class fusion_4(nn.Module):
    """
    simplified version of fusion 2
    """
    def __init__(self, bn = False):
        print("USING FUSION 4")
        super(fusion_4, self).__init__()
        self.fusion_layer = nn.Sequential(Conv2d(3, 8, 3, same_padding=True, bn=False),
                                    Conv2d(8, 1, 1, same_padding=True, bn=False))

    def forward(self, current_level, next_level, image = "", alpha_image = 1./255.):
        features = torch.cat((current_level, next_level, image*alpha_image), dim = 1)
        den = self.fusion_layer(features)
        return den

class evaluator(nn.Module):
    '''
    Sub - Network to weight outputs of different levels in DEMO quad net
    '''
    def __init__(self, input_channels = 96):
        super(evaluator, self).__init__()
        self.classifier =  nn.Sequential(nn.Linear(input_channels, 1),
                                        nn.Sigmoid())
        
    def forward(self, embed):
        embed_feat = F.avg_pool2d(embed, embed.size()[2:]).view(embed.size(0), embed.size(1))
        result = self.classifier(embed_feat)

        return result

class QUADTREE_1(nn.Module):
    '''
    DEMO of Quadtree network for crowd counting
    '''
    
    def __init__(self, levels = 4):
        super(QUADTREE_1, self).__init__()
        self.levels = levels
        self.encoders = ListModule(self, 'encoders_')
        self.decoders = ListModule(self, 'decoders_')
        self.evaluators = ListModule(self, 'evaluators_')
        for i in range(levels):
            self.encoders.append(encoder())
            self.decoders.append(decoder())
        for i in range(levels - 1):
            self.evaluators.append(evaluator())
        
        self.reconstructions = []
        self.visited= [False]*self.levels
        print("quad tree created", len(self.decoders))

    def forward(self, im_data, level, first_time = True):
        if first_time:
            self.visited = [False]*self.levels
            self.reconstructions = []
        if not self.visited[level]:
            self.reconstructions.append([])
            self.visited[level] = True

        _, _, w, h = im_data.size()
        x, x1, x2, x3 = self.encoders[level](im_data)
        reconstruction = self.decoders[level](x, x1, x2, x3)
        reconstruction = F.upsample(reconstruction, size = (w, h), mode = 'bilinear')

        #print("in level", level, "data size", im_data.size(), w, h)
        if level < self.levels - 1:
            weight = self.evaluators[level](x3)
            #separate quadtree

            hh = h if h % 2 == 0 else h - 1
            ww = w if w % 2 == 0 else w - 1

            im_data_1 =  im_data[:, :, 0:ww//2, 0:hh//2]
            im_data_2 =  im_data[:, :, 0:ww//2, hh//2:hh]
            im_data_3 =  im_data[:, :, ww//2:ww, 0:hh//2]
            im_data_4 =  im_data[:, :, ww//2:ww, hh//2:hh]
            
            den_1 = self.forward(im_data_1, level + 1, False)
            den_2 = self.forward(im_data_2, level + 1, False)
            den_3 = self.forward(im_data_3, level + 1, False)
            den_4 = self.forward(im_data_4, level + 1, False)

            next_level = torch.empty((1, 1, ww, hh))
            next_level[:, :, 0:ww//2, 0:hh//2] = den_1
            next_level[:, :, 0:ww//2, hh//2:hh] = den_2
            next_level[:, :, ww//2:ww, 0:hh//2] = den_3
            next_level[:, :, ww//2:ww, hh//2:hh]= den_4
            next_level = F.upsample(next_level, size = (w, h), mode = 'bilinear')
            next_level = next_level.cuda()
            density_map = (1. - weight)*reconstruction + weight*next_level
        else:
            density_map = reconstruction
        #print("end level", level, density_map.size(), im_data.size())
        self.reconstructions[level].append(density_map)
        return density_map

import time
class QUADTREE_2_1(nn.Module):
    '''
    Quadtree network for crowd counting using independent encoder, evaluator and decoder for each level
    '''
    
    def __init__(self, levels = 4):
        super(QUADTREE_2_1, self).__init__()
        self.levels = levels
        self.fusion = fusion()
        self.encoders = ListModule(self, 'encoders_')
        self.decoders = ListModule(self, 'decoders_')
        self.evaluators = ListModule(self, 'evaluators_')
        for i in range(levels):
            self.encoders.append(encoder())
            self.decoders.append(decoder())
        for i in range(levels - 1):
            self.evaluators.append(evaluator())
            
        
        self.reconstructions = []
        self.discriminators = []
        self.upsamples = []
        self.visited= [False]*self.levels
        print("using quadTree Net with NOT shared weights")

    def forward(self, im_data, level, first_time = True):
        if first_time:
            self.visited = [False]*self.levels
            self.reconstructions = []
            self.upsamples = []
            self.discriminators = []
        if not self.visited[level]:
            self.reconstructions.append([])
            self.discriminators.append([])
            self.upsamples.append([])
            self.visited[level] = True

        batch_size, _, h, w = im_data.size()
        start = time.time()
        x, x1, x2, x3 = self.encoders[level](im_data)
        end = time.time()
        elapsed = end - start
        #if level == 0:
            #print ("\tencoder {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
        start = time.time()
        reconstruction = self.decoders[level](x, x1, x2, x3)
        end = time.time()
        elapsed = end - start
        #if level == 0:
            #print ("\tdecoder {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
        self.upsamples[level].append(reconstruction)

        #print("in level", level, "data size", im_data.size(), w, h)
        

        if level < self.levels - 1:
            start = time.time()            
            weight = self.evaluators[level](x3)
            end = time.time()
            elapsed = end - start
            #if level == 0:
                #print ("\tdiscriminator {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
            self.discriminators[level].append(weight)
            #separate quadtree
            start = time.time()            
            chunks = torch.chunk(im_data, chunks = 2, dim = 2)
            im_data_1, im_data_2 = torch.chunk(chunks[0], chunks = 2, dim = 3)
            im_data_3, im_data_4 = torch.chunk(chunks[1], chunks = 2, dim = 3)

            end = time.time()
            elapsed = end - start
            #if level == 0:
                #print ("\tseparate {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
            
            start = time.time()            
            den_1 = self.forward(im_data_1, level + 1, False)
            den_2 = self.forward(im_data_2, level + 1, False)
            den_3 = self.forward(im_data_3, level + 1, False)
            den_4 = self.forward(im_data_4, level + 1, False)
            end = time.time()
            elapsed = end - start
            #if level == 0:
                #print ("\tforward parts {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
            
            start = time.time()
            next_level = torch.cat((torch.cat((den_1, den_2), dim = 3), torch.cat((den_3, den_4), dim = 3)), dim = 2)
            next_level = next_level.cuda()
            end = time.time()
            elapsed = end - start
            #if level == 0:
                #print ("\tjoin {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))

            start = time.time()
            #print("level", level, reconstruction.size(), next_level.size(), im_data.size(), weight.size())
            reconstruction_weighted = ((1. - weight)*reconstruction.view(batch_size, -1)).view(batch_size, 1, h, w)
            next_level_weighted = ((weight)*next_level.view(batch_size, -1)).view(batch_size, 1, h, w)
            density_map = self.fusion(reconstruction_weighted, next_level_weighted, im_data)
            end = time.time()
            elapsed = end - start
            #if level == 0:
                #print ("\tfusion {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))

        else:
            density_map = reconstruction
        #print("end level", level, density_map.size(), im_data.size())
        self.reconstructions[level].append(density_map)
        return density_map

class QUADTREE_2_2(nn.Module):
    '''
    Quadtree network for crowd counting using shared weights for modules in tree levels
    '''
    
    def __init__(self, levels = 4, fusion = 1):
        super(QUADTREE_2_2, self).__init__()
        self.levels = levels
        if fusion == 1:
            self.fusion = fusion_1()
        elif fusion == 2:
            self.fusion = fusion_2()
        elif fusion == 3:
            self.fusion = fusion_3()
        elif fusion == 4:
            self.fusion = fusion_4()
        else: 
            raise Exception("Error: Invalid fusion layer choice {}".format(fusion))
        self.encoder = encoder()
        self.decoder = decoder()
        self.evaluator = evaluator()
        
        self.reconstructions = []
        self.discriminators = []
        self.upsamples = []
        self.visited= [False]*self.levels
        print("using quadTree Net with shared weights")

    def forward(self, im_data, level, first_time = True):
        if first_time:
            self.visited = [False]*self.levels
            self.reconstructions = []
            self.upsamples = []
            self.discriminators = []
        if not self.visited[level]:
            self.reconstructions.append([])
            self.discriminators.append([])
            self.upsamples.append([])
            self.visited[level] = True

        batch_size, _, h, w = im_data.size()
        start = time.time()
        x, x1, x2, x3 = self.encoder(im_data)
        end = time.time()
        elapsed = end - start
        #if level == 0:
            #print ("\tencoder {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
        start = time.time()
        reconstruction = self.decoder(x, x1, x2, x3)
        end = time.time()
        elapsed = end - start
        #if level == 0:
            #print ("\tdecoder {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
        self.upsamples[level].append(reconstruction)

        #print("in level", level, "data size", im_data.size(), w, h)
        

        if level < self.levels - 1:
            start = time.time()            
            weight = self.evaluator(x3)
            end = time.time()
            elapsed = end - start
            #if level == 0:
                #print ("\tdiscriminator {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
            self.discriminators[level].append(weight)
            #separate quadtree
            start = time.time()            
            chunks = torch.chunk(im_data, chunks = 2, dim = 2)
            im_data_1, im_data_2 = torch.chunk(chunks[0], chunks = 2, dim = 3)
            im_data_3, im_data_4 = torch.chunk(chunks[1], chunks = 2, dim = 3)

            end = time.time()
            elapsed = end - start
            #if level == 0:
                #print ("\tseparate {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
            
            start = time.time()            
            den_1 = self.forward(im_data_1, level + 1, False)
            den_2 = self.forward(im_data_2, level + 1, False)
            den_3 = self.forward(im_data_3, level + 1, False)
            den_4 = self.forward(im_data_4, level + 1, False)
            end = time.time()
            elapsed = end - start
            #if level == 0:
                #print ("\tforward parts {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
            
            start = time.time()
            next_level = torch.cat((torch.cat((den_1, den_2), dim = 3), torch.cat((den_3, den_4), dim = 3)), dim = 2)
            next_level = next_level.cuda()
            end = time.time()
            elapsed = end - start
            #if level == 0:
                #print ("\tjoin {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))

            start = time.time()
            #print("level", level, torch.max(reconstruction), torch.max(next_level), torch.max(im_data))
            reconstruction_weighted = ((1. - weight)*reconstruction.view(batch_size, -1)).view(batch_size, 1, h, w)
            next_level_weighted = ((weight)*next_level.view(batch_size, -1)).view(batch_size, 1, h, w)
            density_map = self.fusion(reconstruction_weighted, next_level_weighted, im_data)
            end = time.time()
            elapsed = end - start
            #if level == 0:
                #print ("\tfusion {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))

        else:
            density_map = reconstruction
        #print("end level", level, density_map.size(), im_data.size())
        self.reconstructions[level].append(density_map)
        return density_map

class QUADTREE_2_3(nn.Module):
    '''
    Quadtree network for crowd counting using shared weights for modules in tree levels + reconstruction loss in intermediate layer of autoencoder
    '''
    
    def __init__(self, levels = 4):
        super(QUADTREE_2_3, self).__init__()
        self.levels = levels
        self.fusion = fusion()
        self.encoder = encoder()
        self.decoder = decoder()
        self.evaluator = evaluator()
        self.intermediate_fusion = nn.Sequential(Conv2d(96, 1, 1, same_padding=True, bn=False))
        
        self.reconstructions = []
        self.discriminators = []
        self.upsamples = []
        self.intermediates = []
        self.visited= [False]*self.levels
        print("using quadTree Net with shared weights + intermediate loss for reconstruction")

    def forward(self, im_data, level, first_time = True):
        if first_time:
            self.visited = [False]*self.levels
            self.reconstructions = []
            self.upsamples = []
            self.discriminators = []
            self.intermediates = []
        if not self.visited[level]:
            self.reconstructions.append([])
            self.discriminators.append([])
            self.upsamples.append([])
            self.intermediates.append([])
            self.visited[level] = True

        batch_size, _, h, w = im_data.size()
        start = time.time()
        x, x1, x2, x3 = self.encoder(im_data)
        end = time.time()
        elapsed = end - start
        #if level == 0:
            #print ("\tencoder {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
        start = time.time()
        reconstruction = self.decoder(x, x1, x2, x3)
        end = time.time()
        elapsed = end - start
        #if level == 0:
            #print ("\tdecoder {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
        self.upsamples[level].append(reconstruction)

        intermediate_autoencoder = self.intermediate_fusion(x3)
        self.intermediates[level].append(intermediate_autoencoder)

        #print("in level", level, "data size", im_data.size(), w, h)
        

        if level < self.levels - 1:
            start = time.time()            
            weight = self.evaluator(x3)
            end = time.time()
            elapsed = end - start
            #if level == 0:
                #print ("\tdiscriminator {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
            self.discriminators[level].append(weight)
            #separate quadtree
            start = time.time()            
            chunks = torch.chunk(im_data, chunks = 2, dim = 2)
            im_data_1, im_data_2 = torch.chunk(chunks[0], chunks = 2, dim = 3)
            im_data_3, im_data_4 = torch.chunk(chunks[1], chunks = 2, dim = 3)

            end = time.time()
            elapsed = end - start
            #if level == 0:
                #print ("\tseparate {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
            
            start = time.time()            
            den_1 = self.forward(im_data_1, level + 1, False)
            den_2 = self.forward(im_data_2, level + 1, False)
            den_3 = self.forward(im_data_3, level + 1, False)
            den_4 = self.forward(im_data_4, level + 1, False)
            end = time.time()
            elapsed = end - start
            #if level == 0:
                #print ("\tforward parts {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
            
            start = time.time()
            next_level = torch.cat((torch.cat((den_1, den_2), dim = 3), torch.cat((den_3, den_4), dim = 3)), dim = 2)
            next_level = next_level.cuda()
            end = time.time()
            elapsed = end - start
            #if level == 0:
                #print ("\tjoin {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))

            start = time.time()
            #print("level", level, torch.max(reconstruction), torch.max(next_level), torch.max(im_data))
            reconstruction_weighted = ((1. - weight)*reconstruction.view(batch_size, -1)).view(batch_size, 1, h, w)
            next_level_weighted = ((weight)*next_level.view(batch_size, -1)).view(batch_size, 1, h, w)
            density_map = self.fusion(reconstruction_weighted, next_level_weighted, im_data)
            end = time.time()
            elapsed = end - start
            #if level == 0:
                #print ("\tfusion {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))

        else:
            density_map = reconstruction
        #print("end level", level, density_map.size(), im_data.size())
        self.reconstructions[level].append(density_map)
        return density_map


class QUADTREE_2_4(nn.Module):
    '''
    Autoencoder of QUADTREE_2_2
    '''
    
    def __init__(self, levels = 4, fusion = 1):
        super(QUADTREE_2_4, self).__init__()

        self.encoder = encoder()
        self.decoder = decoder()
        print("using autoencoder of quadTree Net")

    def forward(self, im_data):

        batch_size, _, h, w = im_data.size()
        start = time.time()
        x, x1, x2, x3 = self.encoder(im_data)
        end = time.time()
        elapsed = end - start
        #if level == 0:
            #print ("\tencoder {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
        start = time.time()
        density_map = self.decoder(x, x1, x2, x3)
        end = time.time()
        elapsed = end - start
        
        return density_map


class QUADTREE_STEPS(nn.Module):
    '''
    Quadtree network for crowd counting using shared weights for modules in tree levels + multiple fusion steps
    '''
    
    def __init__(self, levels = 4, fusion = 1, fusion_steps = 2):
        super(QUADTREE_STEPS, self).__init__()
        self.levels = levels
        self.fusion_steps = fusion_steps
        if fusion == 1:
            self.fusion = fusion_1()
        elif fusion == 2:
            self.fusion = fusion_2()
        elif fusion == 3:
            self.fusion = fusion_3()
        elif fusion == 4:
            self.fusion = fusion_4()
        else: 
            raise Exception("Error: Invalid fusion layer choice {}".format(fusion))
        self.encoder = encoder()
        self.decoder = decoder()
        self.evaluator = evaluator()
        
        self.reconstructions = []
        self.discriminators = []
        self.upsamples = []
        self.visited= [False]*self.levels
        print("using quadTree Net with shared weights + {} fusion steps".format(fusion_steps))

    def forward(self, im_data, level, first_time = True):
        if first_time:
            self.visited = [False]*self.levels
            self.reconstructions = []
            self.upsamples = []
            self.discriminators = []
        if not self.visited[level]:
            self.reconstructions.append([])
            self.discriminators.append([])
            self.upsamples.append([])
            self.visited[level] = True

        batch_size, _, h, w = im_data.size()
        x, x1, x2, x3 = self.encoder(im_data)
        upsampled = self.decoder(x, x1, x2, x3)
        self.upsamples[level].append(upsampled)

        if level < self.levels - 1:
            weight = self.evaluator(x3)
            self.discriminators[level].append(weight)
            #separate quadtree
            chunks = torch.chunk(im_data, chunks = 2, dim = 2)
            im_data_1, im_data_2 = torch.chunk(chunks[0], chunks = 2, dim = 3)
            im_data_3, im_data_4 = torch.chunk(chunks[1], chunks = 2, dim = 3)
            
            den_1_steps = self.forward(im_data_1, level + 1, False)
            den_2_steps = self.forward(im_data_2, level + 1, False)
            den_3_steps = self.forward(im_data_3, level + 1, False)
            den_4_steps = self.forward(im_data_4, level + 1, False)
            
            node_reconstruction_steps = []
            last_reconstruction = upsampled
            for i in range(self.fusion_steps):
                den_1 = den_1_steps[i]
                den_2 = den_2_steps[i]
                den_3 = den_3_steps[i]
                den_4 = den_4_steps[i]

                next_level = torch.cat((torch.cat((den_1, den_2), dim = 3), torch.cat((den_3, den_4), dim = 3)), dim = 2)
                next_level = next_level.cuda()

                last_weighted = ((1. - weight)*last_reconstruction.view(batch_size, -1)).view(batch_size, 1, h, w)
                next_level_weighted = ((weight)*next_level.view(batch_size, -1)).view(batch_size, 1, h, w)
                density_map = self.fusion(last_weighted, next_level_weighted, im_data)
                node_reconstruction_steps.append(density_map.clone())

                last_reconstruction = density_map

        else:
            density_map = upsampled
            node_reconstruction_steps = []
            for i in range(self.fusion_steps):
                node_reconstruction_steps.append(density_map)
        
        self.reconstructions[level].append(node_reconstruction_steps)
        return node_reconstruction_steps

class QUADTREE_3_1(nn.Module):
    '''
    Quadtree network for crowd counting using shared weights for modules in tree levels, iterative implementation with one autoencoder forward pass
    '''
    
    def __init__(self, levels = 4):
        super(QUADTREE_3_1, self).__init__()
        self.levels = levels
        self.fusion = fusion()
        self.encoder = encoder()
        self.decoder = decoder()
        self.evaluator = evaluator()
        
        self.reconstructions = []
        self.discriminators = []
        self.upsamples = []
        self.intermediate = []
        print("using quadTree Net with shared weights and iterative implementation")

    def forward(self, im_data, **args):
        x, x1, x2, x3 = self.encoder(im_data)
        upsampled = self.decoder(x, x1, x2, x3)
        weight = self.evaluator(x3)

        self.upsamples.append([upsampled])
        self.intermediate.append([x3]) 
        self.discriminators.append([weight])
        self.reconstructions.append([])
        img_data_levels = []
        img_data_levels.append([im_data])
        #create upsampled tensors for quad tree
        for i in range(1, self.levels):
            self.upsamples.append([])
            self.intermediate.append([])
            self.discriminator.append([])
            self.reconstructions.append([])
            img_data_levels.append([])
            for img, node, inter in zip(img_data_levels[i - 1], self.upsamples[i - 1], self.intermediate[i - 1]):
                #divide upsampled
                chunks = torch.chunk(node, chunks = 2, dim = 2)
                data_1, data_2 = torch.chunk(chunks[0], chunks = 2, dim = 3)
                data_3, data_4 = torch.chunk(chunks[1], chunks = 2, dim = 3)
                self.upsamples[i].append(data_1)
                self.upsamples[i].append(data_2)
                self.upsamples[i].append(data_3)
                self.upsamples[i].append(data_4)
                if i == self.levels - 1:
                    self.reconstructions[i].append(data_1)
                    self.reconstructions[i].append(data_2)
                    self.reconstructions[i].append(data_3)
                    self.reconstructions[i].append(data_4)

                if i < self.levels - 1:    
                    chunks = torch.chunk(img, chunks = 2, dim = 2)
                    data_1, data_2 = torch.chunk(chunks[0], chunks = 2, dim = 3)
                    data_3, data_4 = torch.chunk(chunks[1], chunks = 2, dim = 3)
                    img_data_levels[i].append(data_1)
                    img_data_levels[i].append(data_2)
                    img_data_levels[i].append(data_3)
                    img_data_levels[i].append(data_4)
                    
                    #divide intermediate encoder layer and compute discriminator score
                    chunks = torch.chunk(inter, chunks = 2, dim = 2)
                    data_1, data_2 = torch.chunk(chunks[0], chunks = 2, dim = 3)
                    data_3, data_4 = torch.chunk(chunks[1], chunks = 2, dim = 3)
                    self.intermediate[i].append(data_1)
                    self.intermediate[i].append(data_2)
                    self.intermediate[i].append(data_3)
                    self.intermediate[i].append(data_4)
                    
                    self.discriminators[i].append(self.evaluator(im_data_1))
                    self.discriminators[i].append(self.evaluator(im_data_2))
                    self.discriminators[i].append(self.evaluator(im_data_3))
                    self.discriminators[i].append(self.evaluator(im_data_4))
            for i in range(self.levels - 2, -1, - 1):
                for j in range(self.upsamples[i]):
                    den_1 = self.reconstructions[i+1][j*4]
                    den_2 = self.reconstructions[i+1][j*4 + 1]
                    den_3 = self.reconstructions[i+1][j*4 + 2]
                    den_4 = self.reconstructions[i+1][j*4 + 3]
                    next_level = torch.cat((torch.cat((den_1, den_2), dim = 3), torch.cat((den_3, den_4), dim = 3)), dim = 2)
                    batch_size, _, h, w = self.upsamples[i][j].size()
                    weight = self.discriminators[i][j]
                    reconstruction_weighted = ((1. - weight)*reconstruction.view(batch_size, -1)).view(batch_size, 1, h, w)
                    next_level_weighted = ((weight)*next_level.view(batch_size, -1)).view(batch_size, 1, h, w)
                    density_map = self.fusion(reconstruction_weighted, next_level_weighted, img_data_levels[i][j])
                    self.reconstructions[i].append(density_map)
            return reconstructions[0][0]


class QUADTREE_MCNN(nn.Module):
    '''
    Quadtree network for crowd counting using shared weights for modules in tree levels
    '''
    
    def __init__(self, levels = 4):
        super(QUADTREE_MCNN, self).__init__()
        self.levels = levels
        self.fusion = fusion()
        self.encoder = MCNN_encoder()
        self.decoder = MCNN_decoder()
        self.evaluator = evaluator(input_channels = 128)
        
        self.reconstructions = []
        self.discriminators = []
        self.upsamples = []
        self.visited= [False]*self.levels
        print("using MCNN quadTree Net with shared weights")

    def forward(self, im_data, level, first_time = True):
        if first_time:
            self.visited = [False]*self.levels
            self.reconstructions = []
            self.upsamples = []
            self.discriminators = []
        if not self.visited[level]:
            self.reconstructions.append([])
            self.discriminators.append([])
            self.upsamples.append([])
            self.visited[level] = True

        batch_size, _, h, w = im_data.size()
        start = time.time()
        x = self.encoder(im_data)
        end = time.time()
        elapsed = end - start
        #if level == 0:
            #print ("\tencoder {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
        start = time.time()
        reconstruction = self.decoder(im_data, x)
        end = time.time()
        elapsed = end - start
        #if level == 0:
            #print ("\tdecoder {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
        self.upsamples[level].append(reconstruction)

        #print("in level", level, "data size", im_data.size(), w, h)
        

        if level < self.levels - 1:
            start = time.time()            
            weight = self.evaluator(x)
            end = time.time()
            elapsed = end - start
            #if level == 0:
                #print ("\tdiscriminator {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
            self.discriminators[level].append(weight)
            #separate quadtree
            start = time.time()            
            chunks = torch.chunk(im_data, chunks = 2, dim = 2)
            im_data_1, im_data_2 = torch.chunk(chunks[0], chunks = 2, dim = 3)
            im_data_3, im_data_4 = torch.chunk(chunks[1], chunks = 2, dim = 3)

            end = time.time()
            elapsed = end - start
            #if level == 0:
                #print ("\tseparate {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
            
            start = time.time()            
            den_1 = self.forward(im_data_1, level + 1, False)
            den_2 = self.forward(im_data_2, level + 1, False)
            den_3 = self.forward(im_data_3, level + 1, False)
            den_4 = self.forward(im_data_4, level + 1, False)
            end = time.time()
            elapsed = end - start
            #if level == 0:
                #print ("\tforward parts {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
            
            start = time.time()
            next_level = torch.cat((torch.cat((den_1, den_2), dim = 3), torch.cat((den_3, den_4), dim = 3)), dim = 2)
            next_level = next_level.cuda()
            end = time.time()
            elapsed = end - start
            #if level == 0:
                #print ("\tjoin {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))

            start = time.time()
            #print("level", level, torch.max(reconstruction), torch.max(next_level), torch.max(im_data))
            reconstruction_weighted = ((1. - weight)*reconstruction.view(batch_size, -1)).view(batch_size, 1, h, w)
            next_level_weighted = ((weight)*next_level.view(batch_size, -1)).view(batch_size, 1, h, w)
            density_map = self.fusion(reconstruction_weighted, next_level_weighted, im_data)
            end = time.time()
            elapsed = end - start
            #if level == 0:
                #print ("\tfusion {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))

        else:
            density_map = reconstruction
        #print("end level", level, density_map.size(), im_data.size())
        self.reconstructions[level].append(density_map)
        return density_map
