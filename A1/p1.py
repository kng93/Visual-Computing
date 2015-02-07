from pylab import *
from PIL import Image # For cropping (?)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
import sys

# Combine the blue, green, and red channels to make the coloured picture
def colour_image(blue, green, red):
    height = blue.shape[0]
    width = blue.shape[1]
    coloured = zeros((height, width, 3)).astype(uint8)
    
    coloured[:,:, 0] = red
    coloured[:,:, 1] = green
    coloured[:,:, 2] = blue
    
    return coloured

# Crop images im1 and im2 based on row i and column j
# im1 is the base image, and im2 is being fitted onto im1
def crop_base(im1, im2, i, j):
    im1H, im1W = im1.shape
    im2H, im2W = im2.shape
    
    if i < 0 and j < 0:
        im1 = im1[:im1H+i, :im1W+j]
        im2 = im2[-i:, -j:]
    elif i < 0 and j >= 0:
        im1 = im1[:im1H+i, j:]
        im2 = im2[-i:, :im2W-j]
    elif i >= 0 and j < 0:
        im1 = im1[i:, :im1W+j]
        im2 = im2[:im2H-i, -j:]
    else:
        im1 = im1[i:, j:]
        im2 = im2[:im2H-i, :im2W-j]
        
    return im1, im2

# Calculate the difference between the two images as im2 is shifted and 
# compared to im1
def calculate_diff(im1, im2, is_SSD):
    # Initialization
    x = 10; lowest = 999999999999; highest = 0
    
    # Changes in the images
    cur1 = im1; cur2 = im2
    row = 0; col = 0
    
    # Shift im2 over im1
    for i in range(-x, x): # Row
        for j in range(-x, x): # Column

            # Crop the matrices to match one another
            cur1, cur2 = crop_base(im1, im2, i, j)
            
            # LOOK AT IT LIKE A FLOAT OR THE CALCULATIONS ARE WRONG
            cur1 = cur1.astype(float)
            cur2 = cur2.astype(float)
            val = 0
            
            # Calculate the SSD            
            if is_SSD:
                val = (np.square(cur2 - cur1)).sum()
                lowest = val if lowest > val else lowest
            else:
                dot_prod = (cur1*cur2).sum()
                norm_calc = norm(cur1)*norm(cur2)
                val = dot_prod / norm_calc
                highest = val if highest < val else highest

            # If it's the closest match
            if (is_SSD and lowest == val) or (not(is_SSD) and highest == val):
                row = i
                col = j

    return row, col 

# Separate the image into 3 images (colour channels) and crop
def separate_images(image):
    height = image.shape[0] / 3
    
    # Get the 3 separate images
    blue = image[:height]
    green = image[height:2*height]
    red = image[2*height:]
    
    # Make sure they're the same height
    redH = red.shape[0]
    blueH = blue.shape[0]
    if redH > blueH:
        diff = redH - blueH
        red = red[:redH-diff,:]
    
    # Crop out the borders from the images
    blueH, blueW = blue.shape
    cropH = blueH * 0.05
    cropW = blueW * 0.05
    
    blue = blue[cropH:-cropH, cropW:-cropW]
    green = green[cropH:-cropH, cropW:-cropW]
    red = red[cropH:-cropH, cropW:-cropW]
    
    return blue, green, red

# Apply the matching to the original image given the scaled information
def apply_orig(blueO, greenO, redO, r1, c1, r2, c2, div):
    r1 = r1 * div
    r2 = r2 * div
    c1 = c1 * div
    c2 = c2 * div
    
    blue1, green1 = crop_base(blueO, greenO, r1, c1)
    red1, extra = crop_base(redO, greenO, r1, c1)
    
    blue1, red1 = crop_base(blue1, red1, r2, c2)
    green1, extra = crop_base(green1, red1, r2, c2)
    
    return blue1, green1, red1

def main(argv): 
    filename = ''
    image = []
    threshold = 1.5; norm = 1024

    # Make sure that the argument is correct
    if len(argv) != 2:
        sys.exit('Usage: run A1.py <filename>')
    else:
        filename = argv[1]
    
    # Try to read the image given the file name
    try:
        image = plt.imread(filename)
    except IOError as e:
        sys.exit("I/O error({0}): {1}".format(e.errno, e.strerror))
    except:
        sys.exit("Unexpected error", sys.exc_info()[0])

    # Measure start time
    start = time.clock()

    # Resize the image if it's too large
    orig_image = imresize(image, image.shape)
    div = image.shape[0] / norm
    if div > threshold:
        image = imresize(image, (image.shape[0]/div, image.shape[1]/div))
    else:
        image = orig_image # Makes sure the values are from 0-255
    
    blue, green, red = separate_images(image)
    if div > threshold:
        origB, origG, origR = separate_images(orig_image)
    
    # Calculate the best match using SSD, crop images to match one another
    r1, c1 = calculate_diff(blue, green, True)
    blueS, greenS = crop_base(blue, green, r1, c1)
    redS, extra = crop_base(red, green, r1, c1)
    
    r2, c2 = calculate_diff(blueS, redS, True)
    blueS, redS = crop_base(blueS, redS, r2, c2)
    greenS, extra = crop_base(greenS, redS, r2, c2)
    
    # If the image is large - use the info from scaled-down image to apply 
    # to the original image
    if div > threshold:
        blueS, greenS, redS = apply_orig(origB, origG, origR, r1, c1, r2, c2, 
        div)
    coloured = colour_image(blueS, greenS, redS)
    
    # Show the coloured figure for SSD
    figure(1); imshow(coloured.astype(uint8))

    # Calculate the best match using NCC + colour image
    r1, c1 = calculate_diff(blue, green, False)
    blueN, greenN = crop_base(blue, green, r1, c1)
    redN, extra = crop_base(red, green, r1, c1)

    r2, c2 = calculate_diff(blueN, redN, False)
    blueN, redN = crop_base(blueN, redN, r2, c2)
    greenN, extra = crop_base(greenN, redN, r2, c2)
    
    # If the image is large - use the info from scaled-down image to apply 
    # to the original image
    if div > threshold:
        blueN, greenN, redN = apply_orig(origB, origG, origR, r1, c1, r2, c2, 
        div)
    coloured = colour_image(blueN, greenN, redN)
    
    # Show the coloured figure for NCC
    figure(2); imshow(coloured.astype(uint8))
    
    # Measure end time
    end = time.clock()
    print end - start
    
    # Without this, the images would hang on CDF and not display
    pause(1)
    
if __name__ == "__main__":
    sys.exit(main(sys.argv))
