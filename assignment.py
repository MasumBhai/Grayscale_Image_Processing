import numpy as np
import cv2


def ideal_low_pass_filter(imagePath, output_image_format='jpeg'):
    img = cv2.imread(imagePath, 0)

    min = np.minimum(img.shape[0], img.shape[1])
    # Convert image to 2D array
    img = cv2.resize(img, (min, min))
    # converting to float value
    pixel_values = np.float32(img)
    print("\ninputted image shape: " + str(pixel_values.shape))

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

    cv2.imwrite('low_pass_output.' + output_image_format, imback)
    return imback


def gaussian_high_pass_filter(imagePath, output_image_format='jpeg'):
    img = cv2.imread(imagePath, 0)

    min = np.minimum(img.shape[0], img.shape[1])
    # Convert image to 2D array
    img = cv2.resize(img, (min, min))
    # converting to float value
    pixel_values = np.float32(img)
    print("\ninputted image shape: " + str(pixel_values.shape))

    M, N = img.shape
    # computing the 2-d fourier transformation of the image
    fourier_image = np.fft.fft2(img)

    u = np.array(range(0, M))
    v = np.array(range(0, N))
    idx = np.where(u > (M / 2))
    u[idx] = u[idx] - M
    idy = np.where(v > N / 2)
    v[idy] = v[idy] - N
    [V, U] = np.meshgrid(v, u)
    D = (U ** 2 + V ** 2) ** (1 / 2)

    cutoff = int(input("Enter Cut-off Frequency: "))

    H = 1 - np.exp(((-1) * (D ** 2)) / (2 * (cutoff ** 2)))
    G = H * fourier_image
    # inverse of the 2-dimensional discrete Fourier Transform
    imback = np.fft.ifft2(G)
    imback = np.uint8(np.real(imback))

    cv2.imwrite('high_pass_output.' + output_image_format, imback)
    return imback


def mean_clustering_to_segment_image(imagePath, output_image_format='jpeg'):
    img = cv2.imread(imagePath, 0)

    min = np.minimum(img.shape[0], img.shape[1])
    # Converting image to 2D array
    img = cv2.resize(img, (min, min))

    # converting to float value
    pixel_values = np.float32(img)
    print("\ninputted image shape: " + str(pixel_values.shape))

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

    cv2.imwrite('segmented_output.' + output_image_format, segmented_image)
    return segmented_image


def separate_objects_by_erosion(imagePath, output_image_format='jpeg'):
    img = cv2.imread(imagePath, 0)

    min = np.minimum(img.shape[0], img.shape[1])
    # Converting image to 2D array
    img = cv2.resize(img, (min, min))

    # converting to float value
    pixel_values = np.float32(img)
    print("\ninputted image shape: " + str(pixel_values.shape))

    # Taking a matrix of size 7 as the kernel
    # kernel_size = 7
    kernel_size = int(input("Enter Kernel Size(1~7) : "))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # performing erosion
    img_erosion = cv2.erode(img, kernel, cv2.BORDER_REFLECT)

    cv2.imwrite('Eroded_image.' + output_image_format, img_erosion)
    return img_erosion


def main():
    path = 'input_images/test_data.jpeg'
    image_format = 'jpeg'

    option = int(
        input(
            "1 : Ideal Low Pass Filter\n2 : Gaussian High Pass Filter\n3 : Mean Cluster to Segment\n4 : Separate Objects by Erosion\nEnter your option(1~4) : "))
    if option == 1:
        output_image = ideal_low_pass_filter(imagePath=path, output_image_format=image_format)
        cv2.imshow('low_pass_output', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif option == 2:
        output_image = gaussian_high_pass_filter(imagePath=path, output_image_format=image_format)
        cv2.imshow('high_pass_output', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif option == 3:
        output_image = mean_clustering_to_segment_image(imagePath=path, output_image_format=image_format)
        cv2.imshow('segmented_output', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif option == 4:
        output_image = separate_objects_by_erosion(imagePath=path, output_image_format=image_format)
        cv2.imshow('Eroded_image', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\nInvalid Option. Please Try Again.\n")
        main()


if __name__ == "__main__":
    main()
