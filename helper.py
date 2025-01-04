import cv2
import numpy as np

def processing_img(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img_rgb = np.repeat(img[..., np.newaxis], 3, axis=-1)
    img_rgb = np.expand_dims(img_rgb, axis=0)
    return img_rgb
