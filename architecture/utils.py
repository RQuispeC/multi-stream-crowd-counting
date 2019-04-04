import cv2
import numpy as np
import os

import torch
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
import matplotlib.pylab as plt
import gc

MAXIMUM_CNT = 2.7675038540969217

def save_results(img, gt_density_map, et_density_map, idx, output_dir, loss = 'MSE', model = 'QUAD'):
    idx = idx[0]
    img = img[0, 0]
    gt_density_map = np.array(gt_density_map[0, 0])
    et_density_map = et_density_map[0, 0].cpu().detach().numpy()
    if loss == 'MSE':
        gt_count = np.sum(gt_density_map)
        et_count = np.sum(et_density_map)
    elif loss == 'KL-divergence':
        gt_density_map = ((gt_density_map))*MAXIMUM_CNT
        et_density_map = ((et_density_map))*MAXIMUM_CNT
        gt_count = np.sum(((gt_density_map)))
        et_count = np.sum(((et_density_map)))
    else:
        raise Exception("invalid loss for plot")
    maxi = gt_density_map.max()
    if maxi != 0:
        gt_density_map = gt_density_map*(255. / maxi)
        et_density_map = et_density_map*(255. / maxi)
    #print("min, max GT - ET", gt_density_map.max(), gt_density_map.min(), et_density_map.max(), et_density_map.min())

    if gt_density_map.shape[1] != img.shape[1]:
        gt_density_map = cv2.resize(gt_density_map, (img.shape[1], img.shape[0]))
        et_density_map = cv2.resize(et_density_map, (img.shape[1], img.shape[0]))
    
    fig = plt.figure(figsize = (30, 20))
    a = fig.add_subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    a.set_title('input')
    plt.axis('off')
    a = fig.add_subplot(1, 3, 2)
    plt.imshow(gt_density_map)
    a.set_title('ground thruth {:.2f}'.format(gt_count))
    plt.axis('off')
    a = fig.add_subplot(1, 3, 3)
    plt.imshow(et_density_map)
    a.set_title('estimated {:.2f}'.format(et_count))
    plt.axis('off')
    
    img_file_name = os.path.join(output_dir, str(idx) + ".jpg")
    fig.savefig(img_file_name, bbox_inches='tight')
    fig.clf()
    plt.close()
    del a
    gc.collect()

def save_density_map(density_map,output_dir, fname='results.png'):    
    density_map = 255*density_map/np.max(density_map)
    density_map= density_map[0][0]
    cv2.imwrite(os.path.join(output_dir,fname),density_map)
    
def display_results(input_img, gt_data,density_map):
    input_img = input_img[0][0]
    gt_data = 255*gt_data/np.max(gt_data)
    density_map = 255*density_map/np.max(density_map)
    gt_data = gt_data[0][0]
    density_map= density_map[0][0]
    if density_map.shape[1] != input_img.shape[1]:
         input_img = cv2.resize(input_img, (density_map.shape[1],density_map.shape[0]))
    result_img = np.hstack((input_img,gt_data,density_map))
    result_img  = result_img.astype(np.uint8, copy=False)
    cv2.imshow('Result', result_img)
    cv2.waitKey(0)

def plot_quad(quadnet, input_img, gt_data, img_id, output_dir, plot_intermediate = False):
    img_id = img_id[0]
    input_img_norm = input_img
    pred_den = quadnet(input_img_norm, gt_data)
    quad_depth = len(quadnet.DME.reconstructions)
    gt_data = torch.tensor(gt_data)
    input_img = torch.tensor(input_img)

    #for each node plot inputs and outputs of fusion layer
    gts_level  = [gt_data]
    in_level = [input_img]
    for level, (rec, disc, dec) in enumerate(zip(quadnet.DME.reconstructions[:-1], quadnet.DME.discriminators[:-1], quadnet.DME.upsamples[:-1])): #ignore leaves
        for ind, (node_rec, node_disc, node_dec) in enumerate(zip(rec, disc, dec)):
            fig = plt.figure(figsize = (30, 20))
            a = fig.add_subplot(2, 4 + plot_intermediate, 1)
            plt.imshow(in_level[ind][0, 0], cmap='gray')
            a.set_title('input')
            plt.axis('off')
            a = fig.add_subplot(2, 4 + plot_intermediate, 2)
            plt.imshow(gts_level[ind][0, 0])
            a.set_title('ground thruth {}'.format(torch.sum(gts_level[ind][0, 0])))
            plt.axis('off')

            if plot_intermediate:
                a = fig.add_subplot(2, 4 + plot_intermediate, 3)
                tmp = quadnet.DME.intermediates[level][ind].clone().cpu().detach().numpy()
                plt.imshow(tmp[0, 0])
                a.set_title('intermediate layer {}'.format(node_disc[0][0]))
                plt.axis('off')

            a = fig.add_subplot(2, 4 + plot_intermediate, 3 + plot_intermediate)
            tmp_dec = node_dec.clone().cpu().detach().numpy()
            plt.imshow(tmp_dec[0, 0])
            a.set_title('decoder disc={}, autoencoder cnt={}'.format(round(node_disc[0][0].item(), 3), round(np.sum(tmp_dec[0, 0]), 3)))
            plt.axis('off')

            a = fig.add_subplot(2, 4 + plot_intermediate, 4 + plot_intermediate)
            tmp_rec = node_rec.clone().cpu().detach().numpy()
            plt.imshow(tmp_rec[0, 0])
            a.set_title('reconstruction {}'.format(np.sum(tmp_rec[0, 0])))
            plt.axis('off')
            for i in range(4):
                a = fig.add_subplot(2, 4 + plot_intermediate, i + 5 + plot_intermediate)
                tmp = quadnet.DME.reconstructions[level + 1][ind*4 + i].clone().cpu().detach().numpy()
                plt.imshow(tmp[0, 0])
                a.set_title('next level {}'.format(i+1))
                plt.axis('off')
            save_path = os.path.join(output_dir, "{}_{}_{}_node.jpg".format(str(img_id).zfill(6), level, str(ind).zfill(3)))
            fig.savefig(save_path, bbox_inches='tight')
            fig.clf()
            plt.close()
            del a
            gc.collect()

        new_gts = []
        new_ins = []
        for img, gt in zip(in_level, gts_level):
            chunks = torch.chunk(gt, chunks = 2, dim = 2)
            gt_1, gt_2 = torch.chunk(chunks[0], chunks = 2, dim = 3)
            gt_3, gt_4 = torch.chunk(chunks[1], chunks = 2, dim = 3)
            
            chunks = torch.chunk(img, chunks = 2, dim = 2)
            img_1, img_2 = torch.chunk(chunks[0], chunks = 2, dim = 3)
            img_3, img_4 = torch.chunk(chunks[1], chunks = 2, dim = 3)

            new_gts.append(gt_1)
            new_gts.append(gt_2)
            new_gts.append(gt_3)
            new_gts.append(gt_4)

            new_ins.append(img_1)
            new_ins.append(img_2)
            new_ins.append(img_3)
            new_ins.append(img_4)
        gts_level = new_gts
        in_level = new_ins
            