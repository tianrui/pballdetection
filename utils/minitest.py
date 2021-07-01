import utilities
import pdb
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def mini_generation():
    """ small test with basic images of two classes of pool balls, with one pose for each
    """
    n = 50
    #objsdir = "~/Documents/pballdetection/assets"
    #bgpath = "~/Documents/pballdetection/bg.png"
    #datadir = "~/Documents/pballdetection/dataset"
    objsdir = "./assets"
    bgpath = "./assets/bg.png"
    datadir = "./dataset"
    
    pdb.set_trace()
    annospath = utilities.compile_img(n, objsdir, bgpath, datadir)

def detectron_test():
    n = 50
    #objsdir = "~/Documents/pballdetection/assets"
    #bgpath = "~/Documents/pballdetection/bg.png"
    #datadir = "~/Documents/pballdetection/dataset"
    objsdir = "./assets"
    bgpath = "./assets/bg.png"
    datadir = "./dataset"
    pdb.set_trace()

    for d in ["train", "val"]:
        DatasetCatalog.register("pball_" + d, lambda d=d: utilities.get_pball_dicts(datadir))
        MetadataCatalog.get("pball_" + d).set(thing_classes=["1"]) # TODO: do we need this?
    pball_metadata = MetadataCatalog.get("pball_train")
    
    from detectron2.engine import DefaultTrainer

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__=="__main__":
    #mini_generation()
    detectron_test()
