import h5py
import pickle
import os
with h5py.File('data/1660.h5','r') as f:
    for key in f.keys():
        print(key)
        print(f[key][...])
        print(f[key].shape)