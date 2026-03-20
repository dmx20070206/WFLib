import numpy as np, os

bg = np.load('/home/dmx/DMX/datasets/Background.npz')
bg_X = bg['X']
bg_y = bg['y']

for name in ['day150', 'day270']:
    out_path = f'/home/dmx/DMX/datasets/OpenWorld/{name}.npz'
    d = np.load(f'/home/dmx/DMX/datasets/OpenWorld_old/{name}.npz')
    X, y = d['X'], d['y']
    mask = y != 102
    X_out = np.concatenate([X[mask], bg_X], axis=0)
    y_out = np.concatenate([y[mask], bg_y], axis=0)
    np.savez(out_path, X=X_out, y=y_out)
    with open('/home/dmx/DMX/build_ow_log.txt', 'a') as f:
        f.write(f'{name}: known={mask.sum()}, total={len(X_out)}, label102={(y_out==102).sum()}\n')

with open('/home/dmx/DMX/build_ow_log.txt', 'a') as f:
    f.write('All done.\n')
