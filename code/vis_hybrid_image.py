import numpy as np
from skimage.transform import resize

def vis_hybrid_image(hybrid_image):
    scales = 5
    scale_factor = 0.5
    padding = 5

    original_height = hybrid_image.shape[0]
    num_colors = hybrid_image.shape[2]
    output = hybrid_image.copy()
    cur_image = hybrid_image.copy()

    for i in range(1, scales):
        output = np.concatenate((output, np.ones((original_height, padding, num_colors))), axis=1)
        cur_image = resize(cur_image, (int(scale_factor*cur_image.shape[0]),
            int(scale_factor*cur_image.shape[1])))
        tmp = np.concatenate((np.ones((original_height - cur_image.shape[0], cur_image.shape[1],
            num_colors)), cur_image), axis=0)
        output = np.concatenate((output, tmp), axis=1)

    return output
