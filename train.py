import os
import os.path as osp
import torch
import numpy as np
import sys
from torch.nn.utils import clip_grad_norm_

from architecture.crowd_count import CrowdCounter
from architecture import network
from architecture.data_loader import ImageDataLoader
from architecture.timer import Timer
from architecture import utils
from architecture.evaluate_model import evaluate_model

import argparse

from manage_data import dataset_loader
from manage_data.utils import Logger, mkdir_if_missing

import time
EPSILON = 1e-10
MAXIMUM_CNT = 2.7675038540969217
NORMALIZE_ADD = np.log(EPSILON) / 2.0

parser = argparse.ArgumentParser(description='Train crowd counting network using data augmentation')
# Datasets
parser.add_argument('-d', '--dataset', type=str, default='ucf',
                    choices=dataset_loader.get_names())
#Data augmentation hyperpameters
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--force-den-maps', action='store_true', help="force generation of dentisity maps for original dataset, by default it is generated only once")
parser.add_argument('--force-augment', action='store_true', help="force generation of augmented data, by default it is generated only once")
parser.add_argument('--displace', default=70, type=int,help="displacement for sliding window in data augmentation, default 70")
parser.add_argument('--size-x', default=256, type=int, help="width of sliding window in data augmentation, default 200")
parser.add_argument('--size-y', default=256, type=int, help="height of sliding window in data augmentation, default 300")
parser.add_argument('--people-thr', default=0, type=int, help="threshold of people sliding window in data augmentation, default 200")
parser.add_argument('--not-augment-noise', action='store_true', help="not use noise for data augmetnation, default True")
parser.add_argument('--not-augment-light', action='store_true', help="not use bright & contrast for data augmetnation, default True")
parser.add_argument('--bright', default=10, type=int, help="bright value for bright & contrast augmentation, defaul 10")
parser.add_argument('--contrast', default=10, type=int, help="contrast value for bright & contrast augmentation, defaul 10")
parser.add_argument('--gt-mode', type=str, default='same', help="mode for generation of ground thruth.")

#preprocess hyperparmeters
parser.add_argument('--clahe', action='store_true', help="use adaptative histogram equalization for preprocessing, default False")
# Optimization options
parser.add_argument('--loss', type=str, default='MSE', help="Loss function used for network: MSE or KL-divergence")
parser.add_argument('--model', type=str, default='MCNN', help="Network model used for training")
parser.add_argument('--fusion-layer', type=int, default=1, help="Fusion layer used for final estimation of QUADNET")
parser.add_argument('--loss-quad', default='all', help="type of loss used for quadtree, values: ['top', 'all', 'cnt', 'mix'], default 'all'")
parser.add_argument('--fusion-steps', default=3, type=int,
                    help="number of times fusion layer is applied in QUAD-TREE model")
parser.add_argument('--max-epoch', default=1200, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    help="initial learning rate")
parser.add_argument('--mm', '--momentum', default=0.9, type=float,
                    help="training momentum")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size (default 32)")
# Miscs
parser.add_argument('--seed', type=int, default=64678, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH', help="root directory where part/fold of previous train are saved")
parser.add_argument('--save-dir', type=str, default='log', help="path where results for each part/fold are saved")
parser.add_argument('--units', type=str, default='', help="folds/parts units to be trained, be default all folds/parts are trained")
parser.add_argument('--augment-only', action='store_true', help="run only data augmentation, default False")
parser.add_argument('--evaluate-only', action='store_true', help="run only data validation, --resume arg is needed, default False")
parser.add_argument('--save-plots', action='store_true', help="save plots of density map estimation (done only in test step), default False")
#parser.add_argument('--eval-step', type=int, default=-1,  help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--debug', action='store_true', help="print infor for gradient debug, default False")
parser.add_argument('--quad-level-stats', action='store_true', help="evaluate mse and mae for each level of quadtree")

args = parser.parse_args()

def train(train_test_unit, out_dir_root):
    output_dir = osp.join(out_dir_root, train_test_unit.metadata['name'])
    mkdir_if_missing(output_dir)
    sys.stdout = Logger(osp.join(output_dir, 'log_train.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    dataset_name = train_test_unit.metadata['name']
    train_path = train_test_unit.train_dir_img
    train_gt_path = train_test_unit.train_dir_den
    val_path =train_test_unit.test_dir_img
    val_gt_path = train_test_unit.test_dir_den

    #training configuration
    start_step = args.start_epoch
    end_step = args.max_epoch
    lr = args.lr
    momentum = args.mm

    #log frequency
    disp_interval = args.train_batch*20

    # ------------
    rand_seed = args.seed
    if rand_seed is not None:
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        torch.cuda.manual_seed(rand_seed)

    # load net
    net = CrowdCounter(loss = args.loss, model = args.model, loss_quad = args.loss_quad, fusion_layer = args.fusion_layer, fusion_steps = args.fusion_steps)
    if not args.resume :
        network.weights_normal_init(net, dev=0.01)
    else:
        #network.weights_normal_init(net, dev=0.01) #init all layers in case of partial net load
        if args.resume[-3:] == '.h5':
            pretrained_model = args.resume
        else:
            resume_dir = osp.join(args.resume, train_test_unit.metadata['name'])
            pretrained_model = osp.join(resume_dir, 'best_model.h5')
        network.load_net(pretrained_model, net)
        print('Will apply fine tunning over', pretrained_model)
    net.cuda()
    net.train()

    params = list(net.parameters())
    params_p = list(filter(lambda p: p.requires_grad, net.parameters()))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

    # training
    train_loss = 0
    step_cnt = 0
    re_cnt = False
    t = Timer()
    t.tic()

    #preprocess flags
    use_clahe = True if args.clahe else False
    gt_downsample = True if not args.model.startswith('QUAD') and args.model != 'UNET' and args.model != 'AUTOENCODER' else False
    quad_level_stats = True if args.quad_level_stats else False
    multiple_size  = True if args.model == 'QUAD-MID' else False
    plot_intermediates = True if args.model == 'QUAD-MID' else False

    data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=gt_downsample, pre_load=False, use_clahe = use_clahe, batch_size = args.train_batch, multiple_size = multiple_size)
    data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=gt_downsample, pre_load=False, use_clahe = use_clahe, batch_size = 1)
    best_mae = sys.maxsize

    for epoch in range(start_step, end_step+1):
        step = 0
        train_loss = 0
        for blob in data_loader:
            step = step + args.train_batch
            im_data = blob['data']
            gt_data = blob['gt_density']
            if args.model == 'QUAD-MID':
                gt_data_small = blob['gt_density_small']

            if args.loss == 'MSE':
                pass
            elif args.loss == 'KL-divergence':
                #print("max", gt_data.max())
                gt_data = gt_data / MAXIMUM_CNT #normalize
                #assert gt_data.max() <= 1, "Invalid GT data, there may be an error in MAXIMUM_CNT"
                #gt_data += 1 #for numerical stability
                
                gt_data += EPSILON #for numerical stability
                #gt_data = np.log(gt_data) #normalize
                pass
            else:
                raise Exception("invalid loss function")
            idx_data = blob['idx']
            im_data_norm = im_data
            if args.model == 'QUAD-MID':
                density_map = net(im_data_norm, gt_data = gt_data, gt_data_small =  gt_data_small, epoch = epoch)
            else:
                density_map = net(im_data_norm, gt_data = gt_data, epoch = epoch)
            loss = net.loss
            train_loss += loss.data.item()
            step_cnt += 1
            optimizer.zero_grad()
            loss.backward()

            #clip_grad_norm_(net.parameters(), 1e-1)

            if args.debug and step_cnt%100 == 0:
                print("gt_data min, max, mean", gt_data.min(), gt_data.max(), gt_data.mean())
                print("dt_data min, max, mean", density_map.min().data.item(), density_map.max().data.item(), density_map.mean().data.item())
                _, _, xx, yy = gt_data.shape
                xx = xx //2
                yy = yy //2
                print("gt =========================== \n", gt_data[0, 0, xx:xx +5, yy: yy+5])
                print("dm =========================== \n", density_map[0, 0, xx:xx +5, yy: yy+5], "\n ===========================")
                len_params = len(list(filter(lambda p: p.requires_grad, net.parameters())))
                for i, param in enumerate(filter(lambda p: p.requires_grad, net.parameters())):
                    if i == 33 or i ==31 or args.model == 'SIMPLE':
                        print(step_cnt, ">>>", loss.data.item(), " **>", i, " -->", param.grad, torch.norm(param.grad).data.item())
            optimizer.step()
            #print(loss.data.item())
            if step % disp_interval == 0:
                duration = t.toc(average=False)
                fps = step_cnt / duration
                density_map = density_map.data.cpu().numpy()
                if args.loss == 'MSE':
                    gt_count = np.sum(gt_data.reshape(args.train_batch, -1), axis = 1)
                    et_count = np.sum(density_map.reshape(args.train_batch, -1), axis = 1)
                elif args.loss == 'KL-divergence':
                    gt_count = np.sum(((gt_data))*MAXIMUM_CNT)
                    et_count = np.sum(((density_map ))*MAXIMUM_CNT)
                else:
                    raise Exception("invalid loss function")
                
                if args.save_plots:
                    start = time.time()

                    plot_save_dir = osp.join(output_dir, 'plot-results-train/')
                    mkdir_if_missing(plot_save_dir)
                    utils.save_results(im_data, gt_data, density_map, idx_data, plot_save_dir, loss = args.loss)
                    if args.model.startswith('QUAD'):
                        print("plotting levels of quadtrees")
                        utils.plot_quad(net, im_data, gt_data, idx_data, plot_save_dir, plot_intermediate = plot_intermediates)

                    end = time.time()
                    elapsed = end - start
                    #print ("plot {}:{}:{}".format(elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60))
                    
                print("epoch: {0}, step {1}/{5}, Time: {2:.4f}s, gt_cnt: {3:.4f}, et_cnt: {4:.4f}, mean_diff: {6:.4f}".format(epoch, step, 1./fps, gt_count[0],et_count[0], data_loader.num_samples, np.mean(np.abs(gt_count - et_count))))
                re_cnt = True    
        
            if re_cnt:
                t.tic()
                re_cnt = False

        #if (epoch % 2 == 0):
        save_name = os.path.join(output_dir, '{}_{}_{}.h5'.format(train_test_unit.to_string(), dataset_name,epoch))
        network.save_net(save_name, net)

        #calculate error on the validation dataset 
        mae,mse = evaluate_model(save_name, data_loader_val, loss_quad = args.loss_quad, model = args.model, loss = args.loss, epoch = epoch, quad_level_stats = quad_level_stats, fusion_layer = args.fusion_layer, fusion_steps = args.fusion_steps)
        if mae < best_mae:
            best_mae = mae
            best_mse = mse
            best_model = '{}_{}_{}.h5'.format(train_test_unit.to_string(),dataset_name,epoch)
            network.save_net(os.path.join(output_dir, "best_model.h5"), net)

        print("Epoch: {0}, MAE: {1:.4f}, MSE: {2:.4f}, loss: {3:.4f}".format(epoch, mae, mse, train_loss))
        print("Best MAE: {0:.4f}, Best MSE: {1:.4f}, Best model: {2}".format(best_mae, best_mse, best_model))

def test(train_test_unit, out_dir_root):
    output_dir = osp.join(out_dir_root, train_test_unit.metadata['name'])
    mkdir_if_missing(output_dir)
    sys.stdout = Logger(osp.join(output_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    dataset_name = train_test_unit.metadata['name']
    val_path =train_test_unit.test_dir_img
    val_gt_path = train_test_unit.test_dir_den

    if not args.resume :
        pretrained_model = osp.join(output_dir, 'best_model.h5')
    else:
        if args.resume[-3:] == '.h5':
            pretrained_model = args.resume
        else:
            resume_dir = osp.join(args.resume, train_test_unit.metadata['name'])
            pretrained_model = osp.join(resume_dir, 'best_model.h5')
    print("Using {} for testing.".format(pretrained_model))

    #preprocess flags
    quad_level_stats = True if args.quad_level_stats else False
    use_clahe = True if args.clahe else False
    gt_downsample = True if not args.model.startswith('QUAD') and args.model != 'UNET' and args.model != 'AUTOENCODER' else False
    plot_intermediates = True if args.model == 'QUAD-MID' else False

    data_loader = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=gt_downsample, pre_load=False, use_clahe = use_clahe)
    mae,mse = evaluate_model(pretrained_model, data_loader, loss_quad = args.loss_quad, model = args.model, loss = args.loss, quad_level_stats = quad_level_stats, fusion_layer = args.fusion_layer, fusion_steps = args.fusion_steps)

    print("MAE: {0:.4f}, MSE: {1:.4f}".format(mae, mse))

    if args.save_plots:
        print("Plotting results for test")
        net = CrowdCounter(model = args.model, loss = args.loss, loss_quad = args.loss_quad, fusion_layer = args.fusion_layer, fusion_steps = args.fusion_steps)
        network.load_net(pretrained_model, net)
        net.cuda()
        net.eval()
        plot_save_dir = osp.join(output_dir, 'plot-results-test/')
        mkdir_if_missing(plot_save_dir)

        for blob in data_loader:
            im_data = blob['data']
            idx_data = blob['idx']
            gt_data = blob['gt_density']
            
            if args.model.startswith('QUAD'):
                utils.plot_quad(net, im_data, gt_data, idx_data, plot_save_dir, plot_intermediate = plot_intermediates)
            else:
                density_map = net(im_data, gt_data = gt_data, epoch = 0)
                utils.save_results(im_data, gt_data, density_map, idx_data, plot_save_dir, loss = args.loss)

def main():
    #augment data

    force_create_den_maps = True if args.force_den_maps else False
    force_augmentation = True if args.force_augment else False
    augment_noise = False if args.not_augment_noise else True 
    augment_light = False if args.not_augment_light else True
    augment_only = True if args.augment_only else False


    dataset = dataset_loader.init_dataset(name=args.dataset
    , force_create_den_maps = force_create_den_maps
    , force_augmentation = force_augmentation
    #sliding windows params
    , gt_mode = args.gt_mode
    , displace = args.displace
    , size_x= args.size_x
    , size_y= args.size_y
    , people_thr = args.people_thr
    #noise_params 
    , augment_noise = augment_noise
    #light_params
    , augment_light = augment_light
    , bright = args.bright
    , contrast = args.contrast)

    if augment_only:
        set_units = [unit.metadata['name'] for unit in dataset.train_test_set]
        print("Dataset train-test units are: {}".format(", ".join(set_units)))
        print("Augment only - network will not be trained")
        return

    metadata = "_".join([args.dataset, dataset.signature(), "{}_{}".format('clahe', args.clahe)])
    out_dir_root = osp.join(args.save_dir, metadata)

    if args.units != '':
        units_to_train = [name.strip() for name in args.units.split(',')]
        set_units = [unit.metadata['name'] for unit in dataset.train_test_set]
        print("Dataset train-test units are: {}".format(", ".join(set_units)))
        set_units = set(set_units)
        for unit in units_to_train:
            if not unit in set_units:
                raise RuntimeError("Invalid '{}' train-test unit".format(unit))
    else:
        units_to_train = [unit.metadata['name'] for unit in dataset.train_test_set]
    units_to_train = set(units_to_train)
    for train_test in dataset.train_test_set:
        if train_test.metadata['name'] in units_to_train:
            if args.evaluate_only:
                print("Testing {}".format(train_test.metadata['name']))
                test(train_test, out_dir_root)
            else:
                print("Training {}".format(train_test.metadata['name']))
                train(train_test, out_dir_root)
                print("Testing {}".format(train_test.metadata['name']))
                test(train_test, out_dir_root)

if __name__ == '__main__':
    main()    

