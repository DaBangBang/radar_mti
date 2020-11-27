import numpy as np
z = np.array([5,6,7,8,9,10])
x = np.array([1,2,3,4,5])
y = np.array([2,3])
x = np.delete(x, y)

print(z[x])