
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import Image



act = ['Aaron Eckhart',  'Adam Sandler',   'Adrien Brody',  'Andrea Anders',    'Ashley Benson',    'Christina Applegate',    'Dianna Agron',  'Gillian Anderson']





def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()            


#Note: you need to create the uncropped folder first in order 
#for this to work

for a in act:
    name = a.split()[1].lower()
    i = 0
    
    # Create a directory for uncropped and cropped images if necessary
    if not os.path.exists("./cropped"):
        os.makedirs("./cropped")
    if not os.path.exists("./uncropped"):
        os.makedirs("./uncropped")
    
    for line in open("faces_subset.txt"):
        if a in line:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            #A version without timeout (uncomment in case you need to 
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
            #timeout is used to stop downloading images which take too long to download
            timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
                
            if not os.path.isfile("uncropped/"+filename):
                continue

            # Modify the image and place into "cropped" directory 
            try:
                im = imread("uncropped/"+filename)
                x1, y1, x2, y2 = line.split()[5].split(',')
                
                # Crop the image, grayscale it, then resize it
                im = im[int(y1):int(y2), int(x1):int(x2)] 
                # Image may already be monochrome
                if (len(im.shape) > 2):
                    im = 0.30*im[:,:,0] + 0.59*im[:,:,1] + 0.11*im[:,:,2]
                im = imresize(im, (32,32))
    
                # Save the image
                im = Image.fromarray(im)
                im.save("cropped/"+filename)
            except:
                e = sys.exc_info()[0]
                print "Error with "+filename+": "+str(e)

            print filename
            i += 1
    
    
