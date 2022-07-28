import numpy as np
import cv2

path = 'test_images/test_data.jpeg'

img = cv2.imread(path, 0)

min = np.minimum(img.shape[0], img.shape[1])
# Convert image to 2D array
img = cv2.resize(img, (min, min))
# converting to float value
pixel_values = np.float32(img)
print("input image shape: " + str(pixel_values.shape))

M, N = img.shape

# computing the 2-d fourier transformation of the image
fourier_image = np.fft.fft2(img)

# ideal low pass filter
u = np.array(range(0, M))
v = np.array(range(0, N))
idx = np.where(u > (M / 2))
u[idx] = u[idx] - M
idy = np.where(v > N / 2)
v[idy] = v[idy] - N
[V, U] = np.meshgrid(v, u)
D = (U ** 2 + V ** 2) ** (1 / 2)

cutoff = int(input("Enter Cut-off Frequency: "))

H = (D <= cutoff)
G = H * fourier_image
# inverse of the 2-dimensional discrete Fourier Transform
imback = np.fft.ifft2(G)
imback = np.uint8(np.real(imback))

# showing result
cv2.imshow('cutoff frequency: ' + str(cutoff), imback)
cv2.imshow('original image', img)

cv2.imwrite('low_pass_output.tif', imback)

cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)
