import models
import numpy as np
import matplotlib.pyplot as plt

g = models.Generator()
z = np.ones((1, 100))

y = g(z)
y = y.numpy()
y = y.reshape((28, 28))
plt.imshow(y, cmap="gray")
plt.show()
