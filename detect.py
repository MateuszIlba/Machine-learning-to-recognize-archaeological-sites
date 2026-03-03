import mrcnn
import sys
#import mrcnn.config
#import mrcnn.model
import mrcnn.visualize
import cv2
import os

ROOT_DIR = "E:/W-I-P/2025-08-24-uczeniemaszynowe-zabytki/uczenie/pouczeniuwyuczone/wkladkauczenia/"
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils


CLASS_NAMES = ['BG','stanowisko']

class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = len(CLASS_NAMES)

model = mrcnn.model.MaskRCNN(mode="inference",
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

model.load_weights(filepath="E:/W-I-P/2025-08-24-uczeniemaszynowe-zabytki/uczenie/pouczeniuwyuczone/wkladkauczenia/test/mask_rcnn_object_0249.h5",
                   by_name=True, exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

image = cv2.imread("E:/W-I-P/2025-08-24-uczeniemaszynowe-zabytki/uczenie/pouczeniuwyuczone/wkladkauczenia/test/0050.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

r = model.detect([image], verbose=0)

r = r[0]

mrcnn.visualize.display_instances(image=image,
                                  boxes=r['rois'],
                                  masks=r['masks'],
                                  class_ids=r['class_ids'],
                                  class_names=CLASS_NAMES,
                                  scores=r['scores'])
