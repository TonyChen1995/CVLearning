import cv2
import numpy as np


def image_crop(img, x, y, w, h): # you code here
    """
    Crop image from (x, y) to (x+w, y+h).
    Parameters
    ----------
    :param img: Input image numpy arrays
    :param x: Initial horizontal coordinates
    :param y: Initial horizontal coordinates
    :param w: Cropping width
    :param h: Cropping height
    """
    
    return img[y:y+h, x:x+w]


def color_shift(img, delta_b=0, delta_g=0, delta_r=0): # you code here
    """
    Shift color in RGB channels.
    Parameters
    ----------
    :param img: Input image numpy arrays
    :param delta_b: Increment in blue channel
    :param delta_g: Increment in green channel
    :param delta_r: Increment in red channel
    """
    
    b, g, r = cv2.split(img)
    newb = np.clip(b + np.full(img.shape[:2], delta_b), 0, 255).astype(img.dtype)
    newg = np.clip(g + np.full(img.shape[:2], delta_g), 0, 255).astype(img.dtype)
    newr = np.clip(r + np.full(img.shape[:2], delta_r), 0, 255).astype(img.dtype)
    return cv2.merge((newb, newg, newr))
    


def rotation(img, center, angle): # you code here
    """
    Image rotation.
    Parameters
    ----------
    :param img: Input image numpy arrays
    :param center: Rotation center
    :param angle: Rotation angle(degrees)
    """
    
    matrix = cv2.getRotationMatrix2D((center[0], center[1]), 30, 1)
    return cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))


def perspective_transform(img, pts1, pts2, out_size=None): # you code here
    """
    Image rotation.
    Parameters
    ----------
    :param img: Input image numpy arrays
    :param pts1: List of 4 points in the input image
    :param pts1: List of 4 points in the output image
    :param out_size: Out size of the output image
    """

    if not out_size:
        out_size = img.shape[:2]
        
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    
    return cv2.warpPerspective(img, matrix, out_size)
