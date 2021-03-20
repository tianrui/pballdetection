import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2, torch
import pandas as pd
import os
""" There are 16 balls in billiards, so 16 classes to detect
"""
def compile_img(n, objsdir, bgpath, size, datadir, labelfile):
    """ Compile images together and save into a dataset at datadir
        Save the answer labels as a compressed numpy array
    """
    bg, objs = load_images(objsdir, bgpath)
    answers = []
    for i in range(n):
        imgi, ansi = generate(img(bg, objs)
        try:
            cv2.imwrite(os.path.join(datadir, "img_"+i+".jpg"))
            answers.append(ansi)
    answers = np.asarray(answers)
    np.savez_compressed(answers)

    return

def generate_img(bg, objs):
    """ Generate a full image with a random set of objects within it
        for a baseline, use 0.8 as the probability of a ball showing up
        
        background image: an opencv(same as a numpy array) matrix

        objs: a list of lists with each row comprising of an object
        each col is an image of the object in a certain pose.
        
        return:
            composed: image with the objects in a random pose overlaid on top
            ans: binary vector for the presence of each object (1 if it is present in the image)
    """
    types,poses = objs.shape[0], objs.shape[1]
    objwidth, objheight = obj[0, 0].shape
    bgwidth, bgheight = bg.shape
    composed = np.deepcopy(bg) # final image

    ans = np.random.choice([0, 1], size=types, replace=True, p=[0.2, 0.8])
    coords = generate_points_with_min_distance(types, bg.shape, objwidth)
    for type, v in enumerate(ans):
        if v == 1:
            poseidx = np.randint(0, poses)
            composed[coords[type][0]:coords[type][0]+objwidth][coords[type][1]:coords[type][1]+objheight] = objs[type][poseidx]

    return composed, ans

def generate_points_with_min_distance(n, shape, min_dist):
    # compute grid shape based on number of points
    width_ratio = shape[1] / shape[0]
    num_y = np.int32(np.sqrt(n / width_ratio)) + 1
    num_x = np.int32(n / num_y) + 1

    # create regularly spaced points
    x = np.linspace(0., shape[1]-1, num_x, dtype=np.float32)
    y = np.linspace(0., shape[0]-1, num_y, dtype=np.float32)
    coords = np.stack(np.meshgrid(x, y), -1).reshape(-1,2)

    # compute spacing
    init_dist = np.min((x[1]-x[0], y[1]-y[0]))

    # perturb points
    max_movement = (init_dist - min_dist)/2
    noise = np.random.uniform(low=-max_movement,
                                high=max_movement,
                                size=(len(coords), 2))
    coords += noise

    return coords.astype(int)

def load_images(objsdir, bgpath):
    """ iterate through every subdirectory containing object images and compile them into an array
        objsdir: path containing a directory for each object type
                 under each object type contains images of different poses of the object
    """
    list_subfolders_with_paths = [[f.path, f.name] for f in os.scandir(objsdir) if f.is_dir()]
    objs = []
    # iterate over each object's directory
    for objpath, objname in list_subfolders_with_paths:
        imgnames = [f.name for f in os.scandir(objpath)]
        objs.append([cv2.imread(os.path.join(objpath, imgname)) for imgname in imgnames])
    bg = cv2.imread(bgpath)
    return bg, objs


