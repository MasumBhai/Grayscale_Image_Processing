import numpy as np
import cv2

path = 'test_images/test_data.jpeg'

img = cv2.imread(path, 0)

min = np.minimum(img.shape[0], img.shape[1])
# Converting image to 2D array
img = cv2.resize(img, (min, min))

# converting to float value
pixel_values = np.float32(img)
print("input image shape: " + str(pixel_values.shape))

# iteration termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)

# number of clusters (K)
k = int(input("Enter Cluster size (1~255) : "))

_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# converting data into 8-bit values
centers = np.uint8(centers)
# flatten the labels array
labels = labels.flatten()

# converting all pixels to the color of the centroids
segmented_image = centers[labels.flatten()]

# reshape back to the original image dimension
segmented_image = segmented_image.reshape(img.shape)

# showing result
cv2.imshow('segmented image ', segmented_image)
cv2.imshow('original image', img)

cv2.imwrite('segmented_output.jpeg', segmented_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)
