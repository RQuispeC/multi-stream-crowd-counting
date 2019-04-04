import numpy as np
import cv2
import os
import random
import pandas as pd


class ImageDataLoader():
    def __init__(self, data_path, gt_path, shuffle=False, batch_size = 1):
        self.data_path = data_path
        self.gt_path = gt_path
        self.batch_size = batch_size
        self.data_files = [filename for filename in os.listdir(data_path) \
                           if os.path.isfile(os.path.join(data_path,filename)) and os.path.splitext(filename)[1] == '.jpg']
        self.data_files.sort()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.shuffle = shuffle
        if shuffle:
            random.seed(2468)
        self.num_samples = len(self.data_files)
        self.blob_list = {}        
        self.id_list = np.arange(0,self.num_samples)
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.data_files)
        files = np.array(self.data_files)
        id_list = np.array(self.id_list)
       
        for ind in range(0, len(id_list), self.batch_size):
            idx = id_list[ind: ind + self.batch_size]
            fnames = files[idx]
            imgs = []
            dens = []
            dens_small = []
            for fname in fnames:
                if not os.path.isfile(os.path.join(self.data_path,fname)):
                    print("Error: file '{}' doen't exists".format(os.path.join(self.data_path,fname)))
                img = cv2.imread(os.path.join(self.data_path,fname),0)
                img = img.astype(np.float32, copy=False)
                img = img.reshape((1,img.shape[0],img.shape[1]))
                
                den = np.load(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.npy'))
                den  = den.astype(np.float32, copy=False)
                ht = img.shape[1]
                wd = img.shape[2]
                wd_1 = (int)(wd/4)
                ht_1 = (int)(ht/4)
                den = cv2.resize(den,(wd_1,ht_1))
                den = den * ((wd*ht)/(wd_1*ht_1)) #fix people count
                
                den = den.reshape((1, den.shape[0], den.shape[1]))
                imgs.append(img)
                dens.append(den)

            blob = {}
            blob['data']=np.array(imgs)
            blob['gt_density']=np.array(dens)
            blob['fname'] = np.array(fnames)
            blob['idx'] = np.array(idx)
            yield blob
            
    def get_num_samples(self):
        return self.num_samples
                
        
            
        
