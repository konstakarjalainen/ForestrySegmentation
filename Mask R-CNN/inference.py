import torch
import os, cv2
import numpy as np
from tqdm import tqdm

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils import logger
from detectron2.data import detection_utils as utils
from detectron2.utils.visualizer import ColorMode, Visualizer

from mask2former.config import add_maskformer2_config
from utils.utils import filter_instances_with_score, get_metadata_from_annos_file

#########################
### PROGRAM VARIABLES ###
#########################

OUTPUT_FOLDER = "outputs/log"              # Training outputs to use for inference
DIRECTORY = '/home/konsta/PycharmProjects/Segmentation/Mask R-CNN/data/prescaled/images/000033.png'     # Directory from which to read image to predict
ANNOTATION = '/home/konsta/PycharmProjects/Segmentation/Mask R-CNN/data/prescaled/coco_annotation.json'     # Annotation file for metadata
DETECTION_THRESHOLD = 0.7                              # Minimal network confidence to keep instance

#########################



if __name__ == "__main__":

    print('GPU available :', torch.cuda.is_available())
    print('Torch version :', torch.__version__, '\n')
    logger = logger.setup_logger(name=__name__)

    # Configure Model
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    cfg.merge_from_file(os.path.join(OUTPUT_FOLDER, "config.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(OUTPUT_FOLDER, "model_final.pth")

    # Create Predictor
    predictor = DefaultPredictor(cfg)
    
    # Run inference on selected folder

    # Load image and metadata
    filepath = os.path.join(DIRECTORY)
    image = utils.read_image(filepath, "BGR")
    metadata = get_metadata_from_annos_file(os.path.join(OUTPUT_FOLDER, ANNOTATION))

    # Run network on image
    outputs = predictor(image)
    instances = filter_instances_with_score(outputs["instances"].to("cpu"), DETECTION_THRESHOLD)

    # Visualize image
    visualizer = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1, instance_mode=ColorMode.SEGMENTATION)
    predictions = visualizer.draw_instance_predictions(instances)
    cv2.imshow('Predictions (ESC to quit)', predictions.get_image()[:, :, ::-1])
    k = cv2.waitKey(0)

    # exit loop if esc is pressed
    if k == 27:
        cv2.destroyAllWindows()


