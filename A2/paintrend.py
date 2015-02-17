import os

###########################################################################
## Handout painting code.
###########################################################################
from PIL import Image
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
import matplotlib.image as mpimg
import scipy as sci
import scipy.misc
from scipy.signal import convolve2d as conv
import canny as can

np.set_printoptions(threshold = np.nan)  

def colorImSave(filename, array):
    imArray = scipy.misc.imresize(array, 3., 'nearest')
    if (len(imArray.shape) == 2):
        scipy.misc.imsave(filename, cm.jet(imArray))
    else:
        scipy.misc.imsave(filename, imArray)

def markStroke(mrkd, p0, p1, rad, val):
    # Mark the pixels that will be painted by
    # a stroke from pixel p0 = (x0, y0) to pixel p1 = (x1, y1).
    # These pixels are set to val in the ny x nx double array mrkd.
    # The paintbrush is circular with radius rad>0
    
    sizeIm = mrkd.shape
    sizeIm = sizeIm[0:2];
    nx = sizeIm[1]
    ny = sizeIm[0]
    p0 = p0.flatten('F')
    p1 = p1.flatten('F')
    rad = max(rad,1)
    # Bounding box
    concat = np.vstack([p0,p1])
    bb0 = np.floor(np.amin(concat, axis=0))-rad
    bb1 = np.ceil(np.amax(concat, axis=0))+rad
    # Check for intersection of bounding box with image.
    intersect = 1
    if ((bb0[0] > nx) or (bb0[1] > ny) or (bb1[0] < 1) or (bb1[1] < 1)):
        intersect = 0
    if intersect:
        # Crop bounding box.
        bb0 = np.amax(np.vstack([np.array([bb0[0], 1]), np.array([bb0[1],1])]), axis=1)
        bb0 = np.amin(np.vstack([np.array([bb0[0], nx]), np.array([bb0[1],ny])]), axis=1)
        bb1 = np.amax(np.vstack([np.array([bb1[0], 1]), np.array([bb1[1],1])]), axis=1)
        bb1 = np.amin(np.vstack([np.array([bb1[0], nx]), np.array([bb1[1],ny])]), axis=1)
        # Compute distance d(j,i) to segment in bounding box
        tmp = bb1 - bb0 + 1
        szBB = [tmp[1], tmp[0]]
        q0 = p0 - bb0 + 1
        q1 = p1 - bb0 + 1
        t = q1 - q0
        nrmt = np.linalg.norm(t)
        [x,y] = np.meshgrid(np.array([i+1 for i in range(int(szBB[1]))]), np.array([i+1 for i in range(int(szBB[0]))]))
        d = np.zeros(szBB)
        d.fill(float("inf"))
        
        if nrmt == 0:
            # Use distance to point q0
            d = np.sqrt( (x - q0[0])**2 +(y - q0[1])**2)
            idx = (d <= rad)
        else:
            # Use distance to segment q0, q1
            t = t/nrmt
            n = [t[1], -t[0]]
            tmp = t[0] * (x - q0[0]) + t[1] * (y - q0[1])
            idx = (tmp >= 0) & (tmp <= nrmt)
            if np.any(idx.flatten('F')):
                d[np.where(idx)] = abs(n[0] * (x[np.where(idx)] - q0[0]) + n[1] * (y[np.where(idx)] - q0[1]))
            idx = (tmp < 0)
            if np.any(idx.flatten('F')):
                d[np.where(idx)] = np.sqrt( (x[np.where(idx)] - q0[0])**2 +(y[np.where(idx)] - q0[1])**2)
            idx = (tmp > nrmt)
            if np.any(idx.flatten('F')):
                d[np.where(idx)] = np.sqrt( (x[np.where(idx)] - q1[0])**2 +(y[np.where(idx)] - q1[1])**2)

            #Pixels within crop box to paint have distance <= rad
            idx = (d <= rad)
        #Mark the pixels
        if np.any(idx.flatten('F')):
            xy = (bb0[1]-1+y[np.where(idx)] + sizeIm[0] * (bb0[0]+x[np.where(idx)]-2)).astype(int)
            sz = mrkd.shape
            m = mrkd.flatten('F')
            m[xy-1] = val
            mrkd = m.reshape(mrkd.shape[0], mrkd.shape[1], order = 'F')

            '''
            row = 0
            col = 0
            for i in range(len(m)):
                col = i//sz[0]
                mrkd[row][col] = m[i]
                row += 1
                if row >= sz[0]:
                    row = 0
            '''
            
            
            
    return mrkd

def paintStroke(canvas, x, y, p0, p1, colour, rad):
    # Paint a stroke from pixel p0 = (x0, y0) to pixel p1 = (x1, y1)
    # on the canvas (ny x nx x 3 double array).
    # The stroke has rgb values given by colour (a 3 x 1 vector, with
    # values in [0, 1].  The paintbrush is circular with radius rad>0
    sizeIm = canvas.shape
    sizeIm = sizeIm[0:2]
    idx = markStroke(np.zeros(sizeIm), p0, p1, rad, 1) > 0
    # Paint
    if np.any(idx.flatten('F')):
        canvas = np.reshape(canvas, (np.prod(sizeIm),3), "F")
        xy = y[idx] + sizeIm[0] * (x[idx]-1)
        canvas[xy-1,:] = np.tile(np.transpose(colour[:]), (len(xy), 1))
        canvas = np.reshape(canvas, sizeIm + (3,), "F")
    return canvas

# Convert image to monochrome using Litwinowicz's suggested intensity
def convert_monochrome(img):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    monochrome = 0.30*r + 0.59*g + 0.11*b
    
    return monochrome

# Get the matrix of theta values corresponding to the gradient
def get_theta(mono_img):
    imin = mono_img.copy() * 255.0
    wsize = 5
    gausskernel = can.gaussFilter(4, window = wsize)
    
    fx = can.createFilter([0,  1, 0,
                            0,  0, 0,
                            0, -1, 0])
    fy = can.createFilter([ 0, 0, 0,
                            1, 0, -1,
                            0, 0, 0])

    imout = conv(imin, gausskernel, 'valid')
    gradxx = conv(imout, fx, 'valid')
    gradyy = conv(imout, fy, 'valid')

    gradx = np.zeros(mono_img.shape)
    grady = np.zeros(mono_img.shape)
    padx = (imin.shape[0] - gradxx.shape[0]) / 2.0
    pady = (imin.shape[1] - gradxx.shape[1]) / 2.0
    gradx[padx:-padx, pady:-pady] = gradxx
    grady[padx:-padx, pady:-pady] = gradyy
    
    # Net gradient is the square root of sum of square of the horizontal
    # and vertical gradients

    grad = hypot(gradx, grady)
    theta = arctan2(grady, gradx)
    theta = 180 + (180 / pi) * theta
    # Only significant magnitudes are considered. All others are removed
    xx, yy = where(grad < 5)
    theta[xx, yy] = 0
    grad[xx, yy] = 0
    
    colorImSave('theta.png', theta)
    # The angles are quantized. This is the first step in non-maximum
    # supression. Since, any pixel will have only 4 approach directions.
    x0,y0 = where(((theta<22.5)+(theta>157.5)*(theta<202.5)
                   +(theta>337.5)) == True)
    x45,y45 = where( ((theta>22.5)*(theta<67.5)
                      +(theta>202.5)*(theta<247.5)) == True)
    x90,y90 = where( ((theta>67.5)*(theta<112.5)
                      +(theta>247.5)*(theta<292.5)) == True)
    x135,y135 = where( ((theta>112.5)*(theta<157.5)
                        +(theta>292.5)*(theta<337.5)) == True)

    theta = theta
    theta[x0,y0] = 0
    theta[x45,y45] = 45
    theta[x90,y90] = 90
    theta[x135,y135] = 135
    
    return theta

# Get the endpoints after accounting for edgels
def getEndpoints(c, delta, len1, len2, canny_im):
    p0, p1 = (c - delta * len2, c + delta * len1)
    t = delta
    csize = canny_im.shape
    
    if t[0] == 0 or t[1] == 0:
        return p0, p1
    # Pixels moving horizontally
    elif abs(t[0]) >= abs(t[1]):
        s = [coord / t[0] for coord in t]
    # Pixels moving vertically
    else:
        s = [coord / t[1] for coord in t]
    
    # Going one direction...
    x = c
    k = 0
    while norm(x-c) <= len1:
        k += 1
        x = c + [round(k*s[0]), round(k*s[1])]
        if csize[0] <= x[1] or csize[1] <= x[0] or \
        canny_im[x[1]-1, x[0]-1]:
            p0 = x
            break
    
    # Going the other direction...
    x = c
    k = 0
    while norm(x-c) <= len2:
        k += 1
        x = c - [round(k*s[0]), round(k*s[1])]
        if csize[0] <= x[1] or csize[1] <= x[0] or \
        canny_im[x[1]-1, x[0]-1]:
            p1 = x
            break

    return p0, p1

# Add random perturbation to colour and intensity
def random_colour(colour):
    ran_inten = np.random.uniform(0.85, 1.15)
    r, g, b = colour[0][0], colour[1][0], colour[2][0]
    
    # Modify the colour
    for col in colour:
        ran_col = (np.random.uniform(-15,15))/255.0
        col[0] = (col[0] + ran_col)*ran_inten
        
        # Make sure it doesn't go off the colour scale
        if col[0] > 1:
            col[0] = 1
        elif col[0] < 0:
            col[0] = 0
    
    return colour

# Add random perturbation to theta
def random_theta(theta):
    ran_theta = np.random.uniform(-15, 15)

    theta = theta + ran_theta
    # Make sure it doesn't get below 0 (max can get up to is 135+15=150)
    if theta < 0:
        theta = 180+theta
    
    return theta


if __name__ == "__main__":
    # Read image and convert it to double, and scale each R,G,B
    # channel to range [0,1].
    imRGB = array(Image.open('orchid.jpg'))
    
    # Get monochrome image
    mono_im = convert_monochrome(imRGB)
    canny_im = can.canny(mono_im, 2.0, 7500, 2000)
    theta_im = get_theta(mono_im)
    
    imRGB = double(imRGB) / 255.0
    plt.clf()
    plt.axis('off')
    
    sizeIm = imRGB.shape
    sizeIm = sizeIm[0:2]
    # Set radius of paint brush and half length of drawn lines
    rad = 3
    halfLen = 5
    
    # Set up x, y coordinate images, and canvas.
    [x, y] = np.meshgrid(np.array([i+1 for i in range(int(sizeIm[1]))]), np.array([i+1 for i in range(int(sizeIm[0]))]))
    canvas = np.zeros((sizeIm[0],sizeIm[1], 3))
    canvas.fill(-1) ## Initially mark the canvas with a value out of range.
    # Negative values will be used to denote pixels which are unpainted.
    
    # Random number seed
    np.random.seed(29645)
    
    # Set the default vector from center to one end of the stroke.
    default_theta = 2 * pi * np.random.rand(1,1)[0][0]
    
    time.time()
    time.clock()
    
    idx = 0;
    negative_pixels = np.where(canvas == -1)
    # While there is at least one negative pixel, loop
    while len(negative_pixels[0]):
        # Finding a random negative pixel
        cntr = np.floor(np.random.rand(1,1)[0][0]*len(negative_pixels[0]))
        cntr = np.array([negative_pixels[1][cntr], negative_pixels[0][cntr]])
        cntr = np.amin(np.vstack((cntr, np.array([sizeIm[1], sizeIm[0]]))), axis=0)
        cntr = np.array([cntr[0]+1, cntr[1]+1])
 
        # Grab colour from image at center position of the stroke.
        colour = np.reshape(imRGB[cntr[1]-1, cntr[0]-1, :],(3,1))
        colour = random_colour(colour)
        # Add the stroke to the canvas
        nx, ny = (sizeIm[1], sizeIm[0])
        
        # Check if center is on an edgel or at the edge; if so, length = 0
        csize = canny_im.shape
        if csize[0] <= cntr[1]-1 or csize[1] <= cntr[0]-1 or \
        canny_im[cntr[1]-1, cntr[0]-1]:
            length1, length2 = (0, 0)
        else:
            length1, length2 = (halfLen, halfLen)      
        
        # Orientation of paint brush strokes; if theta = 0, set to default
        if theta_im[cntr[1]-1, cntr[0]-1] > 0:
            theta = theta_im[cntr[1]-1, cntr[0]-1]+(pi/2)
        else:
            theta = default_theta
        theta = random_theta(theta)
        
        # Set vector from center to one end of the stroke.
        delta = np.array([cos(theta), sin(theta)])
        
        # Get the endpoints
        p0, p1 = getEndpoints(cntr, delta, length1, length2, canny_im)
        canvas = paintStroke(canvas, x, y, p0, p1, colour, rad)
        print 'stroke', idx
        
        negative_pixels = np.where(canvas == -1)
        idx += 1
        
    print "done!"
    print default_theta
    time.time()
    
    canvas[canvas < 0] = 0.0
    plt.clf()
    plt.axis('off')
    plt.imshow(canvas)
    plt.pause(3)
    colorImSave('output.png', canvas)