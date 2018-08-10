from util import *
from scipy.spatial import distance
import glob

def load_images(files):
    images = []
    for file in files:
        print('reading image: {}'.format(file))
        image = load_hsv_image(file)
        dom_colour = get_dominant_color(image)
        images.append([image,dom_colour])
    return images

def travelling_salesman(images):
    size = len(images)
    # Disatance matrix
    A = np.zeros([size, size])
    for x in range(0, size-1):
        for y in range(0, size-1):
            A[x,y] = distance.euclidean(images[x][1],images[y][1])

    # Nearest neighbouyr algorithm
    path, _ = NN(A, 0)

    # Final Array
    colours_nn = []
    old_image=cv2.cvtColor(images[0][0], cv2.COLOR_HSV2BGR)
    for i in path[1:]:
        old_image = np.hstack((old_image, cv2.cvtColor(images[i][0], cv2.COLOR_HSV2BGR)))
    return old_image

def step_sorting(images):
    images.sort(key=lambda image: step(image[1][0],image[1][1],image[1][2],8))
    old_image=cv2.cvtColor(images[0][0], cv2.COLOR_HSV2BGR)
    for image in images[1:]:
        old_image = np.hstack((old_image, cv2.cvtColor(image[0], cv2.COLOR_HSV2BGR)))
    return old_image

def step (h,s,v, repetitions=1):
	lum = 0.5 * v  * (2 - s)

	h2 = int(h * repetitions)
	lum2 = int(lum * repetitions)
	v2 = int(v * repetitions)

	if h2 % 2 == 1:
		v2 = repetitions - v2
		lum = repetitions - lum

	return (h2, lum, v2)
    



files = glob.glob('*.jpg')
images = load_images(files)
image = travelling_salesman(images)
#show results to screen
cv2.imshow('Order', image)
cv2.imwrite('order - Travelling Salesman.png',image)

