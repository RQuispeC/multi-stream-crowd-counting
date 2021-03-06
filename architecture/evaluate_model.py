from architecture.crowd_count import CrowdCounter
import architecture.network as network
from manage_data.utils import mkdir_if_missing as mkdir_if_missing
from architecture.utils import save_results as save_results
import numpy as np
import torch

def evaluate_model(trained_model, data_loader, save_test_results = False, plot_save_dir = "/tmp/"):
    net = CrowdCounter()
    network.load_net(trained_model, net)
    net.cuda()
    net.eval()
    mae = 0.0
    mse = 0.0

    for blob in data_loader:                        
        im_data = blob['data']
        gt_data = blob['gt_density']
        idx_data = blob['idx']
        im_data_norm = im_data / 255.0
        density_map = net(im_data_norm, gt_data)
        density_map = density_map.data.cpu().numpy()
        gt_count = np.sum(gt_data)
        et_count = np.sum(density_map)
        mae += abs(gt_count-et_count)
        mse += ((gt_count-et_count)*(gt_count-et_count))

        if save_test_results:
            print("Plotting results")
            mkdir_if_missing(plot_save_dir)
            save_results(im_data, gt_data, density_map, idx_data, plot_save_dir)
    mae = mae/data_loader.get_num_samples()
    mse = np.sqrt(mse/data_loader.get_num_samples())
    return mae,mse
