from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
import cv2 #for resizing image

def get_dominant_color(image, k=4, image_processing_size = None):
    """
    takes an image as input
    returns the dominant color of the image as a list
    
    dominant color is found by running k means on the 
    pixels & returning the centroid of the largest cluster

    processing time is sped up by working with a smaller image; 
    this resizing can be done with the image_processing_size param 
    which takes a tuple of image dims as input

    >>> get_dominant_color(my_image, k=4, image_processing_size = (25, 25))
    [56.2423442, 34.0834233, 70.1234123]
    """
    #resize image if new dims provided
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size, 
                            interpolation = cv2.INTER_AREA)
    
    #reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    #cluster and assign labels to the pixels 
    clt = KMeans(n_clusters = k)
    labels = clt.fit_predict(image)

    #count labels to find most popular
    label_counts = Counter(labels)

    #subset out most popular centroid
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

    return list(dominant_color)

# By Luke https://stackoverflow.com/a/17507369
def NN(A, start):
    """Nearest neighbor algorithm.
    A is an NxN array indicating distance between N locations
    start is the index of the starting location
    Returns the path and cost of the found solution
    """
    path = [start]
    cost = 0
    N = A.shape[0]
    mask = np.ones(N, dtype=bool)  # boolean values indicating which 
                                   # locations have not been visited
    mask[start] = False

    for i in range(N-1):
        last = path[-1]
        next_ind = np.argmin(A[last][mask]) # find minimum of remaining locations
        next_loc = np.arange(N)[mask][next_ind] # convert to original location
        path.append(next_loc)
        mask[next_loc] = False
        cost += A[last, next_loc]

    return path, cost

def load_hsv_image(file, width=100, height=None):
    image = cv2.imread(file)
    if height==None:
        h, w, _ = image.shape
        height = (width*h)/w
    image = cv2.resize(image, (80,100), interpolation = cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

if __name__=='__main__':
    bgr_image = cv2.resize(cv2.imread('windy tree.jpg'), (100,125), 
                            interpolation = cv2.INTER_AREA)
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    
    
    dom_color = get_dominant_color(hsv_image)

    #create a square showing dominant color of equal size to input image
    dom_color_hsv = np.full(bgr_image.shape, dom_color, dtype='uint8')
    #convert to bgr color space for display
    dom_color_bgr = cv2.cvtColor(dom_color_hsv, cv2.COLOR_HSV2BGR)

    #concat input image and dom color square side by side for display
    output_image = np.hstack((bgr_image, dom_color_bgr))

    #show results to screen
    cv2.imshow('Image Dominant Color', output_image)
    
    
