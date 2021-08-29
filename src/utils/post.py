import numpy as np

import torch 

from multiprocessing import Pool

import tqdm 

import cv2

N_WORKERS = 10

WATERSHED_MORPH_NITER = 2
WATERSHED_DILATE_NITER = 10
WATERSHED_FG_MULT = 0.2

# https://www.youtube.com/watch?v=lOZDTDOlqfk
# https://docs.opencv.org/4.5.2/d3/db4/tutorial_py_watershed.html
def w_watershed(img):

    #Threshold image to binary using OTSU. ALl thresholded pixels will be set to 255
    ret1, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Morphological operations to remove small noise - opening
    #To remove holes we can use closing
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = WATERSHED_MORPH_NITER)

    # let us start by identifying sure background area
    # dilating pixes a few times increases cell boundary to background. 
    # This way whatever is remaining for sure will be background. 
    #The area in between sure background and foreground is our ambiguous area. 
    #Watershed should find this area for us. 
    sure_bg = cv2.dilate(opening,kernel,iterations=WATERSHED_DILATE_NITER)

    # Finding sure foreground area using distance transform and thresholding
    #intensities of the points inside the foreground regions are changed to 
    #distance their respective distances from the closest 0 value (boundary).
    #https://www.tutorialspoint.com/opencv/opencv_distance_transformation.htm
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)

    #Let us threshold the dist transform by starting at 1/2 its max value.
    ret2, sure_fg = cv2.threshold(dist_transform, WATERSHED_FG_MULT*dist_transform.max(),255,0)

    #Later you may realize that 0.2*max value may be better. Also try other values. 
    #High value like 0.7 will drop some small mitochondria. 

    # Unknown ambiguous region is nothing but bkground - foreground
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    #Now we create a marker and label the regions inside. 
    # For sure regions, both foreground and background will be labeled with positive numbers.
    # Unknown regions will be labeled 0. 
    #For markers let us use ConnectedComponents. 
    ret3, markers = cv2.connectedComponents(sure_fg)

    #One problem rightnow is that the entire background pixels is given value 0.
    #This means watershed considers this region as unknown.
    #So let us add 10 to all labels so that sure background is not 0, but 10
    markers = markers+10

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    #Now we are ready for watershed filling. 
    markers = cv2.watershed(img, markers)

    img_w = img
    img_w[markers == -1] = 1

    return img_w

def post_watershed(x):

    x = torch.nn.Sigmoid()(x)
    x = (x > 0.5) * 1

    x = x.numpy().astype(np.uint8)

    indexes = []
    images = []
    with Pool(processes=N_WORKERS) as p:
        with tqdm(total=len(x)) as pbar:
            for idx, result in enumerate(p.imap_unordered(func=w_watershed, iterable=x)):
                indexes.append(idx)
                images.append(result)
                pbar.update()

    images = images[indexes] # Ordenar

    """
    l = []
    for s, img in enumerate(x):
        print(s)
        l.append(w_watershed(img))
    
    """
    return torch.tensor(images)