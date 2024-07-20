import matplotlib.pyplot as plt
import numpy as np
import os

print(os.getcwd())


# Data for plotting from .text file
# datax = np.loadtxt('1.55-LCPx.txt')
datax = np.loadtxt('1.55-LCPx.txt')

datay = np.loadtxt('1.55-LCPy.txt')
datau = np.loadtxt('1.55-LCPu.txt')
datav = np.loadtxt('1.55-LCPv.txt')
print("run")

# Since X and Y define a grid, ensure they are 2D arrays of the same shape
X, Y = np.meshgrid(datax, datay)
print('1')
# Create the streamplot
plt.streamplot(X, Y, datau, datav, density=2, linewidth=1, arrowsize=1, arrowstyle='->')
plt.title('Streamplot from .tex Data')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()
# save the figure
plt.savefig('streamplot.png')
print('2')