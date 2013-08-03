'''
Compare images for similarity using the scikit-image structural similarity test
'''

import os
import sys
from skssim.ssim import structural_similarity as ssim
from skssim.dtype import img_as_float

import numpy as np

import Image, ImageMath
import ImageChops
import math


fname1 = "imageA.jpeg"
fname2 = "imageB.jpeg"

if len(sys.argv) > 2:
    fname1 = sys.argv[1]
    fname2 = sys.argv[2]


# PIL image difference
# abs( image2 - image1 )
# does not account for perceptual similarity

img = Image.open(fname1)
img2 = Image.open(fname2)
diff = ImageChops.difference(img2, img)
box = diff.getbbox()
if box:
    box_size = (box[2] - box[0]) * (box[3] - box[1])
else:
    box_size = 0
print "PIL difference bounding box is %d pixels." % box_size


def image_entropy(img):
    """calculate the entropy of an image
    http://brainacle.com/calculating-image-entropy-with-python-how-and-why.html

    same thing using numpy was slower running locally:
    http://stackoverflow.com/questions/5524179/how-to-detect-motion-between-two-pil-images-wxpython-webcam-integration-exampl
    """
    w, h = img.size
    histogram = img.histogram()
    histogram_length = sum(histogram)

    samples_probability = [float(h) / histogram_length for h in histogram]

    return -sum([p * math.log(p, 2) for p in samples_probability if p != 0])

i_pdiff = image_entropy(diff)
print "PIL image difference %f" % i_pdiff


# SSIM comparison
# Structural Similarity Index
# http://scikit-image.org/docs/dev/api/skimage.measure.html#structural-similarity
# http://scikit-image.org/docs/dev/auto_examples/plot_ssim.html#example-plot-ssim-py

img = img_as_float(img)
img2 = img_as_float(img2)

if img.size > img2.size:
    img2 = np.resize(img2, (img.shape[0], img.shape[1]))
    img = np.resize(img, (img.shape[0], img.shape[1]))
elif img.size < img2.size:
    img = np.resize(img, (img2.shape[0], img2.shape[1]))
    img2 = np.resize(img2, (img2.shape[0], img2.shape[1]))
else:
    img2 = np.resize(img2, (img.shape[0], img.shape[1]))
    img = np.resize(img, (img2.shape[0], img2.shape[1]))


def mse(x, y):
    return np.mean((x.astype(float) - y) ** 2)

i_mse = mse(img, img2)
print "MSE %f" % i_mse

i_ssim = ssim(img, img2)  # , dynamic_range=img2.max() - img2.min())
print "SSIM Structural Similarity %f" % i_ssim

