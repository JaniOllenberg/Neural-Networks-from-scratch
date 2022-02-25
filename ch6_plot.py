import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
X,y = spiral_data(samples=100, classes=3)
plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap='brg')
plt.show()

import numpy
for _ in range(10):
    print(0.05 * numpy.random.randn(2,30))

print(X)
print(y)