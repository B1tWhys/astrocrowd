import rawpy
import imageio
from os import path, listdir, makedirs
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import cv2
import matplotlib.pyplot as plt

def load_image(fname):
    print("opening", fname)
    raw_file = rawpy.imread(fname)
    rgb_file = raw_file.postprocess()
    # rgb_file = map(lambda i: i.postprocess(), raw_files), total=len(fnames)))
    # print(rgb_files[0].shape)
    return rgb_file

def detect_points(image):
    print("detecting points...")
    orb = cv2.ORB_create()
    return orb.detectAndCompute(image, None)

def get_matcher():
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def align_2_images(img1, img2):
    kp1, d1 = detect_points(img1)
    kp2, d2 = detect_points(img2)
    matcher = get_matcher()
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key = lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches)*90)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    height, width, _ = img1.shape
    transformed_img = cv2.warpPerspective(img1, homography, (width, height))
    return transformed_img

def align_2_images_helper(args):
    return align_2_images(*args)

data_dir = '../test_images/photographingspace/raw/'
fnames = listdir(data_dir)
fpaths = [path.join(data_dir, f) for f in fnames]

with Pool() as p:
    print("loading images")
    raw_images = list(tqdm(p.imap(load_image, fpaths), total=len(fnames)))

print("aligning images")
reference = raw_images[0]
images = [reference]
with Pool() as p:
    images += list(tqdm(p.imap(align_2_images_helper, zip(raw_images, [reference]*len(raw_images)))))

# for fname, img in zip(fnames, images)
    

makedirs('./output', exist_ok=True)
avgd = np.sum(images, axis=0) / len(fnames)
print(type(avgd))
# print(avgd.shape)
with open("./output/averaged.png", 'wb') as f:
    imageio.imsave(f, avgd, format='png')

with open("./output/" + fnames[0] + '.png', 'wb') as f:
    imageio.imsave(f, images[-1], format='png')
