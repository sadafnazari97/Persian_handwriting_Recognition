import cv2
import numpy as np
import math
from scipy import ndimage
def preprocess(image, gaussian_kernel):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.GaussianBlur(image, (gaussian_kernel, gaussian_kernel), 0)
	

	_, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	
	
	
	while np.sum(image[0]) == 0:
		image = image[1:]

	while np.sum(image[:,0]) == 0:
		image = np.delete(image,0,1)

	while np.sum(image[-1]) == 0:
		image = image[:-1]

	while np.sum(image[:,-1]) == 0:
		image = np.delete(image,-1,1)
	rows,cols = image.shape
	
	if rows > cols:
		factor = 20.0/rows
		rows = 20
		cols = int(round(cols*factor))
		
		image = cv2.resize(image, (cols,rows))
	else:
		factor = 20.0/cols
		cols = 20
		rows = int(round(rows*factor))
		
		if rows == 0:
			return None
		image = cv2.resize(image, (cols, rows))
	colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
	rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
	image = np.lib.pad(image,(rowsPadding,colsPadding),'constant')
	shiftx,shifty = getBestShift(image)
	image = shift(image,shiftx,shifty)
	# applies preproccesing
	return image
def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty
def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted
