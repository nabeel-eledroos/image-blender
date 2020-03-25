# Image Blender

This python3 script takes two images of equal dimensions and blends them.

To run, just type `python3 code/hybridImage.py `

Python (3.7) Modules required:
- scikit-image
- numpy
- matplotlib
- scipy

Each image can be described as a 'blurry' image + a 'sharp' image. Therefore, we can say that a 'sharp' image is the original image - the 'blurry' one. To produce a blended image we can add one blurry image to another sharp one. This script uses a [Gaussian filter](https://en.wikipedia.org/wiki/Gaussian_blur), which is a kernel of Gaussian values which we use to convolve both images.