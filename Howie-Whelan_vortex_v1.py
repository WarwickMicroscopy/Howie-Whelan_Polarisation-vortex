# -*- coding: utf-8 -*-
"""
A bit of 2-beam diffraction contrast never hurt anyone
Based on Hirsch, Howie, Nicolson, Pashley and Whelan p207

2D plots of intensity with a single vortex along [0,1,0]

v1.0 'multislice' Python and OpenCL 29 April 2021

@author: Richard Beanland, Jon Peters

"""

import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image

import HWV_cl_v1 as hw_cl


# Choose method

save_images = True

toc = time.perf_counter()

#
# # input variables
# |||||||||||||||||||||||||||||||| #

# # material
# Extinction distances
# X0i is the imaginary part of the 000 extinction distance
# thickness fringes disappear at about X0i nm
X0i = 1000.0  # nm
# Xg is the (complex) extinction distance for g
# The imaginary part should be larger than X0i
# now calculated on a voxel-by-voxel basis
# Xg = 100.0+ 1j * X0i * 1.16  # nm
# lattice parameter, nm
a = 0.39
# g-vector, Miller indices
g = ([1, 1, 0])
# electron wavelength, nm
lambda0 = 0.00251
# wave vector magnitude, nm^-1
k = 1/lambda0

# # vortex, runs along x
alpha = 0.6  # determines vortex radius, inversely
# experimentally we see a max strain of 0.018
c0 = 0.03  # vortex magnitude for displacements
# experimentally we see a max displacement of 0.033 nm
c1 = 0.55  # vortex magnitude for polarisation
n = 1.15  # decay power
m = 9  # number of vortices


# # sample and imaging conditions
# crystal thickness, nm
t = 14  # nm
dt = 0.1  # slice thickness, nm


# deviation parameter (typically between -0.1 and 0.1)
s = 0.00  # nm^-1

# x dimension: vertical, in nm
xdim = 101  # nm
# y dimension: horizontal, in nm
ydim = xdim  # nm

# pixel scale is 1 nm per pixel, by default
# We change the effective magnification of the image (more or less pixels)
# by the scale factor pix2nm
# with an according increase (<1) or decrease (>1) in calculation time
pix2nm = 0.2  # nm per pixel

# Gaussian blur sigma, nm
# blursigma = 2.0
#
# |||||||||||||||||||||||||||||||| #
# # end of input


# Set up simulation coords
# list of x-coords
xsiz = int(float(xdim+0.5)/pix2nm)
x = np.zeros((xsiz))
for i in range(xsiz):
    # x runs from -xsiz/2 to +xsiz/2, extra 0.5 places the core between pixels
    x[i] = (float(i+0.5)-float(xsiz)/2)*pix2nm
# list of y-coords
ysiz = int(float(ydim+0.5)/pix2nm)
y = np.zeros((ysiz))
for i in range(ysiz):
    # y runs from -ysiz/2 to +ysiz/2, extra 0.5 places the core between pixels
    y[i] = (float(i+0.5)-float(ysiz/2))*pix2nm
# list of z-coords
zsiz = int(t/dt + 0.5)
z = np.zeros((zsiz))
for i in range(zsiz):
    # z runs from -zsiz/2 to +zsiz/2, extra 0.5 places the core between pixels
    # version with vortices mid-way in the specimen
    z[i] = dt*(float(i+0.5)-float(zsiz/2))
    # version with vortices 7nm below top surface
    # z[i] = dt*(float(i+0.5) - 7.0)
    # version with vortices 7nm above bottom surface
    # z[i] = dt*(float(i+0.5) - zsiz + 7.0)


start_time = time.perf_counter()

# Howie-Whelan calc
cl_hw = hw_cl.ClHowieWhelan()
Ib, Id = cl_hw.calc_2D(xsiz, ysiz, pix2nm, z, k, s, X0i,
                       dt, a, g, c0, c1, alpha, n, m)

end_time = time.perf_counter()
duration = end_time - start_time
print("Main loops took: " + str(duration) + " seconds")


# pixels at 0 and 1 to allow contrast comparisons
Ib[0, 0] = 0
Ib[0, 1] = 1
Id[0, 0] = 0
Id[0, 1] = 1


# Output image display
fig = plt.figure(figsize=(8, 4))
fig.add_subplot(2, 1, 1)
# plt.plot(np.squeeze(Ib[1,:]))
plt.imshow(Ib)
plt.axis("off")
fig.add_subplot(2, 1, 2)
# plt.plot(np.squeeze(Id[1,:]))
plt.imshow(Id)
plt.axis("off")
bbox_inches = 0

# Image saving
if save_images:

    # save & show the result
    imgname = "BF_t=" + str(int(t)) + "_s" + str(s) + ".tif"
    Image.fromarray(Ib).save(imgname)
    imgname = "DF_t=" + str(int(t)) + "_s" + str(s) + ".tif"
    Image.fromarray(Id).save(imgname)

    plotnameP = "t=" + str(int(t)) + "_s" + str(s) + ".png"
    # print(plotnameP)
    plt.savefig(plotnameP)  # , format = "tif")

tic = time.perf_counter()
print("Full function took: " + str(tic - toc) + " seconds")
