import time
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score as BA
import numpy as np
import random
import time
import itertools
import torch.nn.functional as F
import torch
# from PIL import Image
import pytesseract
from torchvision import transforms as T
import yaml

import logging
import logging.config

from warnings import simplefilter
simplefilter(action='ignore', category=UserWarning)

def seed_all(seed):
    # seed everything
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return True


def timing(func):
    def wrapper(args):
        logging.info("Running function {}".format(func.__name__))
        t1 = time.time()
        res = func(args)
        t2 = time.time()
        period = t2 - t1
        logging.info("{} took {} hour {} min {} sec".format(func.__name__, period // 3600, (period % 3600) // 60,
                                                     int(period % 60)))
        return res

    return wrapper

# should rewrite this one
def metric(label, pred):
    pred = torch.argmax(pred, dim=-1)
    cls_report = classification_report(label, pred, digits=4)
    return cls_report


def readConfig(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    return cfg

def saveConfig(path, target):
    # special treatment, logger object should not be dumped
    new_target = {k:v for k,v in target.items() if k != 'logger'}
    with open(path, 'w') as f:
        cfg = yaml.safe_dump(new_target, f)
    return True

def setupLogger(path):
    logging.config.fileConfig(fname=path, disable_existing_loggers=False)
    logger = logging.getLogger(__name__)
    return logger
