import numpy as np

x, x_lens = np.ones((1, 100, 80), dtype=np.float32), np.array([100])
np.save('x.npy', x)
np.save('x_lens.npy', x_lens)
