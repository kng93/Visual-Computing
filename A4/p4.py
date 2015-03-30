import os
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

def main(argv): 
    back1 = []; back2 = []; new_back = []; im1 = []; im2 = []

    # Make sure that the argument is correct
    if len(argv) != 3 and len(argv) != 6:
        sys.exit('Usage: run A1.py <file-segment identifier> <new background> '
        + 'or run A1.py <background1> <background2> <new background> '
        + '<composition1> <composition2>')
    # Set the images
    elif len(argv) == 3:
        try:
            back1 = imread('./imgs/'+argv[1]+'-backA.jpg')/255.0
            back2 = imread('./imgs/'+argv[1]+'-backB.jpg')/255.0
            new_back = imread(argv[2])/255.0
            
            im1 = imread('./imgs/'+argv[1]+'-compA.jpg')/255.0
            im2 = imread('./imgs/'+argv[1]+'-compB.jpg')/255.0
        except IOError as e:
            sys.exit("I/O error({0}): {1}".format(e.errno, e.strerror))
        except:
            sys.exit("Unexpected error", sys.exc_info()[0])
    elif len(argv) == 6:
        try:
            back1 = imread(argv[1])/255.0
            back2 = imread(argv[2])/255.0
            new_back = imread(argv[3])/255.0
            
            im1 = imread(argv[4])/255.0
            im2 = imread(argv[5])/255.0
        except IOError as e:
            sys.exit("I/O error({0}): {1}".format(e.errno, e.strerror))
        except:
            sys.exit("Unexpected error", sys.exc_info()[0])
    
    # Make sure all the images are of the same shape 
    if (back1.shape != back2.shape or back1.shape != new_back.shape or 
    back1.shape != im1.shape or back1.shape != im2.shape):
        sys.exit("Images do not match")
    
    # Create the constant part of the equation
    setup = np.array([[1,0,0,1,0,0],[0,1,0,0,1,0],[0,0,1,0,0,1]])
            
    # Set up the canvas
    height, width, ch = back1.shape
    canvas = np.zeros((height, width, 3))
    foreground = np.zeros((height, width, 3))
    alpha = np.zeros((height, width))
    
    # Iterate over every pixel
    for i in range(height):
        for j in range(width):
            # Set the backgorund and colour info into a vector
            background = hstack((back1[i][j][:3], back2[i][j][:3]))
            colour = hstack((im1[i][j][:3], im2[i][j][:3]))
            
            # Combine the vectors into A and b to solve for x
            A = np.transpose(vstack((setup, -1*background)))
            A_inv = linalg.pinv(A)
            b = np.transpose(colour - background)
            
            x = np.dot(A_inv, b)
            foreground[i][j] = x[0:3]
            canvas[i][j] = foreground[i][j] + (1-x[3])*new_back[i][j][:3]
            alpha[i][j] = x[3]
          
    # Post-processing; clip the images, display, and save them
    canvas = np.clip(canvas, 0, 1)
    foreground = np.clip(foreground, 0, 1)
    alpha = np.clip(alpha, 0, 1)
    figure(1); imshow(canvas)
    figure(2); imshow(alpha)
    figure(3); imshow(foreground)
    
    imsave('./new_comp.jpg', canvas)
    imsave('./foreground.jpg', foreground)
    imsave('./alpha.jpg', alpha)



if __name__ == "__main__":
    sys.exit(main(sys.argv))