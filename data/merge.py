import h5py
import numpy as np
import pickle
l, r = 1, 401
while l < 38119:
    data, mapping, cur = [], {}, 0

    with h5py.File('trainval/%d_%d.h5' % (l, r), 'r') as f:
        data = f['data'][...]
    
    for key in range(l, r):
        with h5py.File('trainval/%d.h5' % int(key), 'r') as f:
            data.append(f['R'][...])
            mapping[key] = cur
            cur += f['R'].shape[0]
        print(key)
    
    with h5py.File('trainval/%d_%d.h5' % (l, r), 'w') as f:
        f['data'] = np.concatenate(data)
    pickle.dump(mapping, open('trainval/%d_%d.pkl' % (l, r), 'wb'))
    print('%d_%d' % (l, r))
    
    for key in range(l, r):
        os.remove('trainval/%d.h5' % int(a[key]['filename'][-10:-4]))

    l = r
    r = min(38119, r + 400)
print('done')

    