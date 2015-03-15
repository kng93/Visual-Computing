
from PIL import Image
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
import scipy as sci
import scipy.misc
from scipy.ndimage import filters
import shutil
import re
import urllib

# Force a new download/cropping of the images
download = False

act = ['Aaron Eckhart',  'Adam Sandler',   'Adrien Brody',  'Andrea Anders',    'Ashley Benson',    'Christina Applegate',    'Dianna Agron',  'Gillian Anderson']

# Gender lists with lower-case last-names of the values in act
fem = ['anders',    'benson',    'applegate',    'agron',  'anderson']
male = ['eckhart',  'sandler',   'brody']

# Globals which may be changed for more testing
nTrain = 100
nValid = 10
nTest = 10
k_set = [2, 5, 10, 20, 50, 80, 100, 150, 200]

# Taken from "get_data.py"; helper function of "get_data" function
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


# Taken from "get_data.py"; Gets the data and crops it
def get_data():
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
            
# Move the files from the begin directory to the end directory
# Helper function for split_set
def mv_files(files, init, nRange, len_act, begin_dir, end_dir):
    
    # If no wrap-around needed for training set
    if init+nRange < len_act:
        for fIdx in range(init, init+nRange):
            shutil.copyfile(begin_dir+"/"+files[fIdx], end_dir+"/"+files[fIdx])
        
        init = init+nRange # Set new init for next partition 
    # Wrap-around needed for training set
    else:
        # Move files from starting index to end
        for fIdx in range(init, len_act):
            shutil.copyfile(begin_dir+"/"+files[fIdx], end_dir+"/"+files[fIdx])
            
        init = nRange-(len_act-init) # Set new init for next partition 
        
        # Move files from beginning to max number (wrapped)
        for fIdx in range(0, init):
            shutil.copyfile(begin_dir+"/"+files[fIdx], end_dir+"/"+files[fIdx])
    
    return init
        

# dir is the directory to extract the images, k is the indicator of where to 
# partitioning. Will take the first 100 from k*10 for training, next 10 for
# validation, next 10 for testing (wraps around to 
def split_set(dir, k):
    # Max k can be 12 (since only guaranteed 100 train, 10 valid, 10 test)
    if k > 12:
        k = k % 12
    
    # Remove previously created directories
    if os.path.exists('./training'):
        shutil.rmtree('./training')
    if os.path.exists('./validation'):
        shutil.rmtree('./validation')
    if os.path.exists('./testing'):
        shutil.rmtree('./testing')
    
    # Create the directories
    os.makedirs("./training")
    os.makedirs("./validation")
    os.makedirs("./testing")
    
    # Get list of each actor's images
    num_act = len(act)
    act_files = []
    for i in range(0, num_act):
        act_files.append([])
    
    # Create list of lists, each nested list contains all of certain actor's names
    for filename in sorted(os.listdir(dir)):
        for i in range(0, num_act):
            if act[i].split()[1].lower() == re.split(r"[0-9]+.", filename)[0]:
                act_files[i].append(filename)
    
    # Move the images to their respective folders (train, valid, test)
    for i in range(0, num_act):
        files = act_files[i]
        init = k*nValid
        len_act = len(act_files[i])
        
        # Move the files to training, validaiton, and testing respectively
        init = mv_files(files, init, nTrain, len_act, dir, "./training")
        init = mv_files(files, init, nValid, len_act, dir, "./validation")
        mv_files(files, init, nTest, len_act, dir, "./testing")


# Get the eigenfaces (code from pca_example)
def get_eigenfaces(faces):
    # get dimensions
    num_data,dim = faces.shape
    
    # center data
    avg = np.mean(faces, axis=0)
    faces = faces - avg
    
    if dim>num_data:
        # PCA - compact trick used
        M = dot(faces, faces.T) # covariance matrix
        e,EV = linalg.eigh(M) # eigenvalues and eigenvectors
        tmp = dot(faces.T,EV).T # this is the compact trick
        V = tmp[::-1] # reverse since last eigenvectors are the ones we want
        S = sqrt(e)[::-1] # reverse since eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        # PCA - SVD used
        U,S,V = linalg.svd(faces)
        V = V[:num_data] # only makes sense to return the first num_data
    
    # return the projection matrix, the variance and the mean
    return V,S,avg


# Save the top x eigenfaces to the directory "./eigenfaces"
def save_top_eigenfaces(eigenfaces, x):
    
    # Delete the directory if it exists so that files aren't mixed
    if os.path.exists('./eigenfaces'):
        shutil.rmtree('./eigenfaces')
    
    os.makedirs('./eigenfaces')
    
    # Save the eigenfaces...
    for i in range(0, x):
        imsave('./eigenfaces/face'+str(i)+'.jpg', eigenfaces[i].reshape([32,32]))
 
 
# Get the face after it has been reduced given the eigenfaces
# Helper function for get_correct_percent
def get_reduced_face(faces, V, k, avg):
    W = np.array([])
    for face in faces:
        # If the face array is empty, give it the first value
        if W.size == 0:
            W = np.asarray([np.dot(V[i,:], (face-avg)) for i in range(k)])
        # Else, append it 
        else:
            temp = np.asarray([np.dot(V[i,:], (face-avg)) for i in range(k)])
            W = vstack((W, temp))

    return W
    

# Check the unknown face against all the reduced-faces to find the greatest match
# Helper function for get_correct_percent
def recognize_face(W, V, k, im, avg):
    face_weight = np.asarray([np.dot(V[i,:], (im-avg)) for i in range(k)])
    least_diff = float("inf")
    lowest_idx = 0
    
    # Iterate over all the faces in the training set to find the closes match
    for i in range(len(W)):
        diff = (norm(W[i] - face_weight) ** 2)
        if least_diff > diff:
            least_diff = diff  
            lowest_idx = i
    
    return lowest_idx


# Get the percentage of correctly identified files in dir
# gender is a toggle to set for gender-identification, or person-identification
# Helper function for run_test
def get_correct_percent(faces, files, V, k, avg, dir, gender):
    
    W = get_reduced_face(faces, V, k, avg)
    correct = 0; total = len(os.listdir(dir))
    fail = 0
    
    # Loop over all the faces in the directory
    for filename in os.listdir(dir):
        im = imread(dir+filename).flatten()
        face_idx = recognize_face(W, V, k, im, avg)
        
        # Get the name of the person (from the file name)
        unk = re.split(r"[0-9]+.", filename)[0]
        match = re.split(r"[0-9]+.", files[face_idx])[0]
        
        # If testing for person, check for exact match
        if (gender == False):
            if (unk == match):
                correct += 1
            # Save failures (only when testing)
            elif (dir == './testing/'):
                shutil.copy2(dir+filename, './mFailure/'+str(fail)+unk+'u.jpg')
                shutil.copy2('./training/'+files[face_idx],
                            './mFailure/'+str(fail)+match+'m.jpg')
                fail += 1
            
        # If testing for gender, check if in global list
        if (gender == True):
            if ((match in male) and (unk in male)) or ((match in fem) and (unk in fem)):
                correct += 1  
            # Save failures (only when testing)
            elif (dir == './testing/'):
                shutil.copy2(dir+filename, './gFailure/'+str(fail)+unk+'u.jpg')
                shutil.copy2('./training/'+files[face_idx],
                            './gFailure/'+str(fail)+match+'m.jpg')
                fail += 1

    return float(correct)/total


# Validate to obtain the parameters, then test on the testing set
def run_test(faces, files, V, k_set, avg, gender):
    best_k = 0; best_percentage = 0
    # Validation step
    for i in range(len(k_set)):
        res = get_correct_percent(faces, files, V, k_set[i], avg, './validation/', gender)
        print "k = ", k_set[i], ", Correct Percent = ", res
        # Keep the k with the best result
        if (best_percentage < res):
            best_k = k_set[i]
            best_percentage = res
    
    # Testing step; run on k with the best result
    return get_correct_percent(faces, files, V, best_k, avg, './testing/', gender)
    

# Compiles the eigenfaces into a single image for the report (taken from pca_example.py)
def display_save_25_comps(V, x, im_shape):
    '''Display 25 components in V'''
    figure()
    for i in range(x):
        plt.subplot(5, 5, i+1)
        plt.axis('off')
        gray()
        imshow(V[i,:].reshape(im_shape))
    savefig('./eigenfaces/display_save_25_comps.jpg')  
    

if __name__ == "__main__":
    # Get the data if doesn't exist, or requested by global "download"
    if (not os.path.exists("./cropped")) or \
    (not os.path.exists("./uncropped")) or download:
        get_data()
        
    gray()
    
    # Value can be changed from 0 to 12 (anything above 12 is modded by 12)
    split_set('./cropped/', 2) 
    files = os.listdir('./training')

    # Put faces into an array
    faces = np.array([])
    for filename in files:
        im = imread('./training/'+filename)
        # Add the first face
        if faces.size == 0:
            faces = im.flatten()
        else:
            faces = vstack((faces, im.flatten()))
    
    # Get the average face and eigenfaces
    V, S, avg = get_eigenfaces(faces)
    
    # Save the eigenfaces and the average face
    save_top_eigenfaces(V, 25)
    display_save_25_comps(V, 25, (32,32))
    imsave('./eigenfaces/avg_face.jpg', avg.reshape([32,32]))
    
    # Delete the directory if it exists so that files aren't mixed
    # Create a directory to put failure cases
    if os.path.exists('./mFailure'): # Failure cases for matching
        shutil.rmtree('./mFailure')
    os.makedirs('./mFailure')
    if os.path.exists('./gFailure'): # Failure cases for gender
        shutil.rmtree('./gFailure')
    os.makedirs('./gFailure')
    
    # Person Recognization
    final = run_test(faces, files, V, k_set, avg, False)
    print "Person: ", final
    
    # Gender Recognization
    final = run_test(faces, files, V, k_set, avg, True)
    print "Gender: ", final

    