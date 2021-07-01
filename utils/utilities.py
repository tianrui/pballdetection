import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2, torch
import pandas as pd
import os, pdb, copy
""" 16 balls in billiards, so 16 classes to detect
"""
def compile_img(n, objsdir, bgpath, datadir):
    """ Compile images together and save into a dataset at datadir
        Save the answer labels as a compressed numpy array
        Input:  n, size of dataset
                objsdir, directory containing objects in subfolders
                bgfile, file of the background image
                datdir, directory to store the generated data in

        Output:  Return the file of the annotations file
                Store the annotations in  the data directory

    """
    bg, objs = load_images(objsdir, bgpath)
    answers = []
    bboxlist = []
    for i in range(n):
        res = generate_img(bg, objs)
        imgi, ansi, bboxi = res
        try:
            cv2.imwrite(str(os.path.join(datadir, "img_"+str(i)+".png")), imgi)
            answers.append(ansi)
            bboxlist.append(bboxi)
        except OSError as err:
            print("Could not write image, OSError: {0}".format(err))
    answers = np.asarray(answers)
    bboxlist = np.asarray(bboxlist, dtype=object)
    np.savez_compressed(os.path.join(datadir, "annotations"), answers=answers, bboxes=bboxlist)
    return os.path.join(datadir, "annotations")

def resize_object_images(bg, objs):
    """ for each object that we have poses for, resize the image to be 1/10 of the shortest
        background side, and keep the aspect ratio. the object images are roughly square in aspect ratios.
        1/10 is an arbitrary selection.
        input is in AxB, background is in CxD, to get the resize to be 1/10 the scaling factor is min of 
        (C/10)/A, (D/10)/A

    """
    bgwidth, bgheight = bg.shape[:2]
    for typeidx, type in enumerate(objs):
        for imageidx, image in enumerate(type):
            scaling = min(bgwidth/(10 * image.shape[0]), bgheight/(10 * image.shape[0]))
            # provide output dimensions as a tuple
            objs[typeidx][imageidx] = cv2.resize(image, (int(scaling * image.shape[0]), int(scaling * image.shape[1])), cv2.INTER_AREA)
    return bg, objs

def generate_img(bg, objs):
    """ Generate a full image with a random set of objects within it
        for a baseline, use 0.8 as the probability of a ball showing up
        
        background image: an opencv(same as a numpy array) matrix

        objs: a list of lists with each row comprising of an object
        each col is an image of the object in a certain pose.
        
        return:
            composed: image with the objects in a random pose overlaid on top
            ans: binary vector for the presence of each object (1 if it is present in the image)
            bboxes: list of bounding boxes with [t, x0, y0, x1, y1] where t is the object type(category)
    """
    bg, objs = resize_object_images(bg, objs)
    types,poses = len(objs), len(objs[0])
    bgwidth, bgheight = bg.shape[:2]
    objwidth, objheight = (objs[0][0]).shape[:2]
    
    #set the object height/width to be 1/10 of the shortest background dimension. No longer necessary with resizing done before
    #objwidth, objheight = min(bg.shape)//10, min(bg.shape)
    composed = copy.deepcopy(bg) # final image

    ans = np.random.choice([0, 1], size=types, replace=True, p=[0.2, 0.8])
    coords_shape = list(bg.shape[:2])
    coords_shape[0] -= objwidth
    coords_shape[1] -= objheight
    coords = generate_points_with_min_distance(types, coords_shape, objwidth)
    bboxes = []
    for type, v in enumerate(ans):
        if v == 1:
            poses = len(objs[type])
            poseidx = np.random.randint(0, poses)

            currentobjwidth, currentobjheight = objs[type][poseidx].shape[:2]
            composed[coords[type][0]:coords[type][0]+currentobjwidth, coords[type][1]:coords[type][1]+currentobjheight][:] = objs[type][poseidx]
            bboxes.append([type, coords[type][0], coords[type][1], coords[type][0]+currentobjwidth, coords[type][1]+currentobjheight])
    return composed, ans, bboxes

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
    # clip the coords to be inside the shape
    clippedx = np.clip(coords[:, 0], 0, shape[0])
    clippedy = np.clip(coords[:, 1], 0, shape[1])
    coords = np.array(list(zip(clippedx, clippedy)))
    return coords.astype(int)

def load_images(objsdir, bgfile):
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
    bg = cv2.imread(bgfile)
    return bg, objs

def get_dataset_for_detectron2(n, datadir):
    """ Register the dataset as a standardized dataset for detectron2
        Input:  n, size of the dataset
                datadir, directory containing images and annotation arrays
        Output: List of dictionaries containing entries of the dataset
        Dictionary fields: file_name, height, width, image_id, annotations{bbox, bbox_mode=XYXY_ABS, category_id}
    """
    annos = np.load(os.path.join(datadir, "annotations"))
    answers, bboxes = annos['answers'], annos['bboxes']
    dataset_dicts = []
    for i in range(n):
        record = {}
        
        filename = os.path.join(datadir, "img_%s"%(str(i))+".jpg")
        height, width = cv2.imread(filename).shape[:2]

        # create list of bboxes according to format
        objs = []
        for idx, box in enumerate(bboxes[i]):
            obj = {
                "category_id": box[0],
                "bbox": [box[1:]],
                "bbox_mode": BoxMode.XYXY_ABS
            }
            objs.append(obj)
        # write all the information into the record entry
        record["file_name"] = filename
        record["height"] = height
        record["width"] = width
        record["annotations"] = objs

        dataset_dicts.append(record)
    return dataset_dicts

def get_pball_dicts(datadir):
    n = 1000
    return get_dataset_for_detectron2(n, datadir)
