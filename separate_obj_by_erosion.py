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

# Taking a matrix of size 7 as the kernel
kernel_size = 7
kernel = np.ones((kernel_size, kernel_size), np.uint8)
# performing erosion
img_erosion = cv2.erode(img, kernel, cv2.BORDER_REFLECT)

# showing result
cv2.imshow('Eroded image ', img_erosion)
cv2.imshow('original image', img)

cv2.imwrite('Eroded_image.jpeg', img_erosion)

cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)
