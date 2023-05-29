import datetime
import torch
import os

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.defaults import DefaultPredictor
from detectron2.evaluation.evaluator import inference_on_dataset
from detectron2.data.datasets import load_coco_json
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.utils import logger
from mask2former.config import add_maskformer2_config
from detectron2 import model_zoo

from utils.utils import *
from utils.custom_trainers import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#########################
### PROGRAM VARIABLES ###
#########################

DO_TRAIN = True              # Whether to train the model
DO_TEST = True               # Whether to test the model 
RESUME = False               # If True, resumes from last_checkpoint

DATASET_FILENAME = '/home/konsta/CanaTree100/annotations/fold_01/train_fold_01.json'          # COCO style annotation file
TESTSET_FILENAME = '/home/konsta/CanaTree100/annotations/fold_01/val_fold_01.json'
IMAGE_DIR = '/home/konsta/CanaTree100/images/'                      # Folder where dataset was downloaded
SPLIT_FRAC = 0.8                                                  # Fraction of data used in train

CONFIG_FILE = "configs/config_standard_maskrcnn.yaml"  # Detectron2 style config file
TEST_WEIGHTS = ""                                       # Checkpoint to use for testing
INITIAL_WEIGHTS = None                                  # Path to a previous checkpoint to finetune or None

SHOW_ANNOTATIONS = False        # Display images and labels before training
SHOW_AUGMENTATIONS = False      # Display augmented images and labels before training

#########################


def init_maskrcnn(config_file, train_dicts, test_dicts, output_dir, fold_num=None, initial_weights=None): 

    # Configure Model - See all parameters here : https://detectron2.readthedocs.io/en/latest/modules/config.html#yaml-config-references
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.OUTPUT_DIR = output_dir
    cfg.DATASETS.TRAIN = ("log_train",)
    cfg.DATASETS.TEST = ("log_test",)
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.MAX_ITER = 70*len(train_dicts)

    # Initialize network weights (to finetune)
    if initial_weights is not None:
        cfg.MODEL.WEIGHTS = initial_weights

    # Rename dataset if using k-fold cross-validation
    if fold_num is not None:
        cfg.DATASETS.TRAIN = [cfg.DATASETS.TRAIN[0] + f"_fold{fold_num}"]
        cfg.DATASETS.TEST = [cfg.DATASETS.TEST[0] + f"_fold{fold_num}"]

    # Initialize Dataloaders
    DatasetCatalog.register(cfg.DATASETS.TRAIN[0], lambda d=cfg.DATASETS.TRAIN[0]: train_dicts)
    MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).set(thing_classes=["log"])
    DatasetCatalog.register(cfg.DATASETS.TEST[0], lambda d=cfg.DATASETS.TEST[0]: test_dicts)
    MetadataCatalog.get(cfg.DATASETS.TEST[0]).set(thing_classes=["log"])

    # Save Model config in output folder
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/config.yaml", "w") as f:
        f.write(cfg.dump())

    # Copy annotation files to output folder (for backups)
    convert_to_coco_json(cfg.DATASETS.TRAIN[0], os.path.join(output_dir, "annos_train.json"))
    convert_to_coco_json(cfg.DATASETS.TEST[0], os.path.join(output_dir, "annos_test.json"))

    return cfg


def train_maskrcnn(cfg, resume=False, show_annos=False, show_augs=False):

    ########################
    #### TRAINING LOGIC ####
    ########################

    # Visualize annotations
    if show_annos:
        dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        visualize_annotations(dicts, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    
    # Visualize data augmentation
    if show_augs:
        train_loader = StandardMaskRCNNTrainer.build_train_loader(cfg)
        visualize_data_augmentation(train_loader, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), image_format=cfg.INPUT.FORMAT)

    # Initialize trainer
    trainer = StandardMaskRCNNTrainer(cfg) 
    trainer.resume_or_load(resume)
    trainer.train() 

    # Save model (use half the space)
    checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.save("model_final")  # save to output folder


def test_maskrcnn(cfg, test_weights):

    #######################
    #### TESTING LOGIC ####
    #######################

    if os.path.isfile(test_weights):
       cfg.MODEL.WEIGHTS = test_weights

    # Create Predictor
    predictor = DefaultPredictor(cfg)
    
    # Evaluate Predictions on Test Dataset
    evaluator = StandardMaskRCNNTrainer.build_evaluator(cfg, cfg.DATASETS.TEST[0])
    val_loader = StandardMaskRCNNTrainer.build_test_loader(cfg, cfg.DATASETS.TEST[0])
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

    # Visualize Predictions
    logs_metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    dicts_test = DatasetCatalog.get(cfg.DATASETS.TEST[0])
    visualize_predictions(predictor, dicts_test, logs_metadata)



if __name__ == "__main__":

    print('GPU available :', torch.cuda.is_available())
    print('Torch version :', torch.__version__, '\n')
    logger.setup_logger(name=__name__)

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    # Init output folder 
    if (RESUME):
        # Take latest run in outputs folder
        output_dir = np.sort([x for x in os.listdir("./outputs/") if os.path.isdir("./outputs/"+x)])[-1]
        output_dir = os.path.join("./outputs", output_dir)
        config_file = os.path.join(output_dir, "config.yaml")
    else:
        output_dir = f'./outputs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}'

    # Split train/test
    train_dicts = np.array(load_coco_json(DATASET_FILENAME, IMAGE_DIR))
    test_dicts = np.array(load_coco_json(TESTSET_FILENAME, IMAGE_DIR))

    cfg = init_maskrcnn(CONFIG_FILE, train_dicts, test_dicts, output_dir, initial_weights=INITIAL_WEIGHTS)

    if DO_TRAIN:
        train_maskrcnn(cfg, RESUME, SHOW_ANNOTATIONS, SHOW_AUGMENTATIONS)
        TEST_WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")       # Ensures trained network will be used for testing 

    if DO_TEST:
        test_maskrcnn(cfg, TEST_WEIGHTS)
