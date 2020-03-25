import numpy as np
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
from vis_hybrid_image import vis_hybrid_image

def imread(path):
    img = plt.imread(path).astype(float)
    #Remove alpha channel if it exists
    if img.ndim > 2 and img.shape[2] == 4:
        img = img[:, :, 0:3]
    #Puts images values in range [0,1]
    if img.max() > 1.0:
        img /= 255.0

    return img

def gaussian(hsize=3,sigma=0.5):
    shape = (hsize, hsize)
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def hybrid_image(im1, im2, sigma1, sigma2):
    im1 = im1.astype(float)
    im2 = im2.astype(float)
    if im1.max() > 1.0:
        im1 /= 255.0
    if im2.max() > 1.0:
        im2 /= 255.0

    filter = gaussian(6*sigma1+1, sigma1)[:,:,np.newaxis]

    im1_gaussian = convolve(im1, filter)
    im2_gaussian = convolve(im2, filter)
    im2_sharp = im2-im2_gaussian
    blend = im1_gaussian + im2_sharp

    np.clip(blend, 0, 1, out = blend)
    return blend

if __name__ == '__main__':
    img1 = imread('data/me.jpg')
    img2 = imread('data/monkey2.jpg')
    #8 and 12
    plt.imshow(vis_hybrid_image(hybrid_image(img1, img2, 8, 12)))
    plt.show()