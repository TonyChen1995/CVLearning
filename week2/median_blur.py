import cv2
import numpy as np
import matplotlib.pyplot as plt


def my_show(img):
    """
    Show opencv image with plt function.
    Parameters
    ----------
    :param img: Input image numpy arrays
    """
    
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    

def sp_noise(image, prob):
    """
    Add salt noise to the input image.
    Parameters
    ----------
    :param img: Input image numpy arrays, dtype should be numpy.uint8.
    """

    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def median_blur(src, ksize=3, border_type=cv2.BORDER_REPLICATE):
    """
    Blurs an image using the median filter.
    The function smoothes an image using the median filter with the 
    ksize × ksize aperture. 
    Each channel of a multi-channel image is processed independently.
    Parameters
    ----------
    :param img: Input image numpy arrays, dtype should be numpy.uint8.
    
    :param ksize: aperture linear size; it must be odd and greater than 1, 
    for example: 3, 5, 7 ....
    
    :param border_type: Pixel extrapolation method. See border types of opencv
    for further help.
    """
    pad_width = ksize // 2
    bottom = cv2.copyMakeBorder(src, pad_width, pad_width, pad_width, pad_width, border_type)
    
    rows, cols, channels = src.shape
    dst = np.empty(src.shape, dtype=np.uint8) 
    
    for i in range(rows):
        for j in range(cols):
            for k in range(channels):
                dst[i][j][k] = np.median(bottom[i:i+ksize, j:j+ksize, k])

    return dst


def my_median(array):
    """
    Calculate the median value of the input array based on counting sort.
    ----------
    :param array: Input image numpy arrays, dtype should be numpy.uint8.
    """
    size = np.size(array)
    max_ele = np.max(array)
    min_ele = np.min(array)
    count = [0] * (max_ele - min_ele + 1)
    for ele in np.nditer(array):
        count[ele-min_ele] += 1
        
    j = 0
    median_idx = size // 2
    for i in range(max_ele - min_ele + 1):
        j += count[i]
        if j >= median_idx:
            break
        
    return i + min_ele
    

def median_blur_faster(src, ksize=3, border_type=cv2.BORDER_REPLICATE):
    
    """
    Blurs an image using the median filter based on counting sort.
    The function smoothes an image using the median filter with the 
    ksize × ksize aperture. 
    Each channel of a multi-channel image is processed independently.
    Parameters
    ----------
    :param img: Input image numpy arrays, dtype should be numpy.uint8.
    
    :param ksize: aperture linear size; it must be odd and greater than 1, 
    for example: 3, 5, 7 ....
    
    :param border_type: Pixel extrapolation method. See border types of opencv
    for further help.
    """
    pad_width = ksize // 2
    bottom = cv2.copyMakeBorder(src, pad_width, pad_width, pad_width, pad_width, border_type)
    
    rows, cols, channels = src.shape
    dst = np.empty(src.shape, dtype=np.uint8) 
    
    for i in range(rows):
        for j in range(cols):
            for k in range(channels):
                
                dst[i][j][k] = my_median(bottom[i:i+ksize, j:j+ksize, k])

    return dst


# read original lenna image 
img = cv2.imread('lenna.jpg', 1)

# resize the image for faster processing
height, width = img.shape[:2]
img = cv2.resize(img, (width//2, height//2), interpolation=cv2.INTER_CUBIC)

# add 3% salt noise
noise_img = sp_noise(img, 0.03)

# call opecv's medianBlur method
cv = cv2.medianBlur(img, 3)

# call my medianBlur method
# it is so slow that it is commmented
# my = median_blur(img, ksize=3, border_type=cv2.BORDER_REPLICATE)

# call my faster medianBlur method based on counting sort
# however it is also slow compared to the above opencv method
my_faster = median_blur_faster(img, ksize=3, border_type=cv2.BORDER_REPLICATE)

# show results
plt.subplot(131)
my_show(noise_img)

plt.subplot(132)
my_show(cv)

plt.subplot(133)
my_show(cv)