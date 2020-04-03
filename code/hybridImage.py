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

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def alignImages(img1, img2, max_shift):
    #Fix img1 red channel
    red_ch_img1 = img1[:, :, 0]
    r1 = red_ch_img1.reshape(-1)

    red_ch_img2 = img2[:, :, 0]
    # green_ch = img[:, :, 1]
    # blue_ch = img[:, :, 2]

    img2_shiftx = 0
    img2_shifty = 0
    min_angle = 1000
    # bshiftx = 0
    # bshifty = 0
    # min2 = 1000
    for x in range(-max_shift[0], max_shift[0]+1):
        for y in range(-max_shift[1], max_shift[1]+1):
            r2 = np.roll(red_ch_img2, [x, y], axis=[0, 1]).reshape(-1)
            ang = angle_between(r1, r2)
            if(ang < min_angle):
                min_angle = ang
                img2_shiftx = x
                img2_shifty = y

    print(img2_shiftx)
    print(img2_shifty)
    img2[:, :, 0] = np.roll(img2[:, :, 0], [img2_shiftx, img2_shifty], axis=[0, 1])
    img2[:, :, 1] = np.roll(img2[:, :, 1], [img2_shiftx, img2_shifty], axis=[0, 1])
    img2[:, :, 2] = np.roll(img2[:, :, 2], [img2_shiftx, img2_shifty], axis=[0, 1])

    # img2 = img2[5:,5:,:]
    # img2 = img2[:-5,:-5,:]

    # final_shift = np.array([[gshiftx, gshifty], [bshiftx, bshifty]])

    return img2


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
    img1 = imread('data/jay2.jpg')
    img2 = imread('data/ye2.jpg')
    max_shift = np.array([15, 15])

    #8 and 12
    # plt.imshow(vis_hybrid_image(hybrid_image(img1, img2, 4, 8))
    img2_aligned = alignImages(img1, img2, max_shift)
    plt.imshow(img2 - img2_aligned)
    # plt.imshow(hybrid_image(img1, img2_aligned, 4, 8))
    # plt.imshow(hybrid_image(img1, img2, 4, 8))
    plt.show()