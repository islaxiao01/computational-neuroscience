import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(7967)

# load data from text file in one line
# documentation for the loadtxt function can be found here
# https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html

# NB there are many possible type of text file you might want to read. 
# data = np.loadtxt('C:/My Documents/TEACHING/NSCI 303/data/Adaline1.dat')

#  replace the following path with the appropriate one for your computer 
file = open('C:/My Documents/TEACHING/NSCI 303/data/Adaline3.dat', 'r')

# the following code will read the file, one line at a time, and stop when it gets to the end

# make the data array large enough for the largest file to be read
data = np.zeros((800, 3))

nline = 0
while True:
#   read one line at a time
    textline = file.readline()
#   test to see if we've reached the end of the file
    if len(textline) == 0: break
#   convert textline to list of numbers
    data[nline, :] = textline.split()
    nline = nline + 1
file.close

# data is an array with values that are the same as those in the file; 3 columns; each
# row gives in order, the x value, the y value and the diagnosis (classification value)

# print the array dimensions
ndims = np.size(data[0, :]) - 1
nvals = nline
print('data array =', nvals,'x',ndims)

# make an array of color values for display - give the colour for each point depending on the diagnosis
colors = []
for i in range (0, nvals):
    if data[i, 2] == 1:
        colors.append('red')
    else:
        colors.append('green')

# find beginnning and ending of X data values, in data[:, 0] - the first column
xmin = np.min(data[:, 0])
xmax = np.max(data[:, 0])

weights = np.zeros(ndims + 1)

# initialise the weights to random values close to zero
sdev = 0.1
for i in range (0, ndims + 1):
    weights[i] = np.random.normal(0., sdev, None)

# coordinates for the decision boundary
x = np.zeros(2)
y = np.zeros(2)

fig, ax = plt.subplots()

# maxitnum can be changed 
maxitnum = 500
# too large a value for gain can stop it converging
gain = 0.0001

#  have an array to keep the weight change
dw = np.zeros(ndims + 1)

for itnum in range (0, maxitnum):
    esum = 0
    dw[:] = 0

    for n in range (0, nvals): 
#       get the output (was z in the lecture)
        o = weights[0]*data[n, 0] + weights[1]*data[n, 1] + weights[2]
#       and the desired output = 2 times (0 or 1) -1 = -1 or +1
        t = 2*data[n, 2] - 1

#       these are the Adaline learning rules (the delta rule, or Widrow-Hoff learning)
        dw[0] = dw[0] + data[n, 0]*(t - o)
        dw[1] = dw[1] + data[n, 1]*(t - o)
#       this is the rule for the bias weight with a fixed input of 1 
        dw[2] = dw[2] + (t - o)
#       keep track of the error, it should decrease at each step
        esum += (t - o)**2 

#   now update all the weights in one go after going through all the data values
    weights += gain*dw 

#   might want to print this out, or use it to decide when to stop
    esum = np.sqrt(esum)

#   calculate beginning and end of straight line (decision boundary)
#   convert from weights to y = mx + c, as shown in lecture slides
    m = -weights[0]/weights[1]
    c = -weights[2]/weights[1]
#   make the line go between min and max x values; this is not always ideal
    x[0] = xmin
    x[1] = xmax
    y[0] = m*x[0] + c
    y[1] = m*x[1] + c

    ax.clear()

#   for the scatter plot: x values are data[0:nvals, 0], y values are data[0:nvals, 1]
#   s is the symbol size; c = array of colours that match the diagnoses
    plt.scatter(data[0:nvals, 0], data[0:nvals, 1], s = 10, c = colors)
#   plot the decision boundary
    plt.plot(x, y, c = 'black')

#   wait or there will be nothing to look at
    plt.pause(.05)

# keep showing the graph after finishing
plt.show()