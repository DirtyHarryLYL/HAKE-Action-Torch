import h5py
import pickle
import os

l, r = 1, 401
while l < 38119:
    with h5py.File('trainval/%d_%d.h5' % (l, r), 'r') as f:
        data = f['data'][...]
    mapping = pickle.load(open('trainval/%d_%d.pkl' % (l, r), 'rb'))

    for key in range(l, r):
        start = mapping[key]
        end = mapping[key+1] if key+1 in mapping else data.shape[0]
        filename = 'trainval/%d.h5' % key
        with h5py.File(filename, 'w') as f:
            f['R'] = data[start:end]
        print(key)

    l = r
    r = min(38119, r + 400)

print('done')