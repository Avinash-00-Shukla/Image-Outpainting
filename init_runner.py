# -*- coding: utf-8 -*-
"""
@author: avina
"""

#%% Section 1
import util
IN_PATH, RSZ_PATH, OUT_PATH = 'DATASET\\raw', 'DATASET\\places_resized', 'DATASET\\places\\all_images.npy'
util.resize_images(IN_PATH, RSZ_PATH)
util.compile_images(RSZ_PATH, OUT_PATH)

#%% Section 2
import numpy as np
data = np.load('DATASET\\places\\all_images.npy')
idx_test = np.random.choice(36500, 100, replace=False)
idx_train = list(set(range(36500)) - set(idx_test))
imgs_train = data[idx_train]
imgs_test = data[idx_test]
np.savez('DATASET\\places\\places_128.npz', imgs_train=imgs_train, imgs_test=imgs_test, idx_train=idx_train, idx_test=idx_test)