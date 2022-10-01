import numpy as np
a = np.array([[1, 2], [3, 5]])
y = np.expand_dims(a, axis=2)
z = np.expand_dims(a, axis=1)
print(a.shape)  # (2,2)
print(y.shape)  # (2, 2, 1)
print(z.shape)  # (2, 1, 2)
