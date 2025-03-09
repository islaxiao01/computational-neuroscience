import numpy as np
import math
import matplotlib.pyplot as plt

def Gaussian(r:float, sigma:float):
    s2 = 2*sigma*sigma
    Gaussian = math.exp(-r**2/s2)
    return Gaussian

image = np.zeros((500, 500))
x0 = 250
y0 = 250
sigma1 = 30
sigma2 = 90

# normalising factors for each Gaussian
E = 1/(2*np.pi*sigma1**2)
I = 1/(2*np.pi*sigma2**2)

for i in range (0, 500):
    for j in range (0,500):
       x = i - x0
       y = j - y0
       r = np.sqrt(x*x + y*y)
       image[i, j] = E*Gaussian(r, sigma1) - I*Gaussian(r, sigma2)

plt.imshow(image, cmap = 'gray')
plt.show()
plt.imshow(image, cmap = 'viridis')
plt.show()
plt.imshow(image, cmap = 'plasma')
plt.show()
plt.imshow(image, cmap = 'hsv')
plt.show()

# plot a 1D cross section to confirm
xvals = list(range(500))
plt.plot(xvals, image[:, 250])
plt.show()
