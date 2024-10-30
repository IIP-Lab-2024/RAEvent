import logging
import torch

from reader.reader import init_dataset, init_formatter, init_test_dataset
from model import get_model
from model.optimizer import init_optimizer
# from .output_init1 import init_output_function
from torch import nn
import json

logger = logging.getLogger(__name__)


def init_all(checkpoint, mode, *args, **params):
    result = {}

    logger.info("Begin to initialize dataset and formatter...")
    if mode == "train":
        result["train_dataset"], result["valid_dataset"] = init_dataset()
    else:
        result["test_dataset"] = init_test_dataset(*args, **params)

    logger.info("Begin to initialize models...")

    model = get_model("pairwise")(mode, *args, **params) # model_name = pairwise
    optimizer = init_optimizer(model, *args, **params)
    trained_epoch = 0
    global_step = 0

    model = model.cuda()
    

    try:
        parameters = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        if hasattr(model, 'module'):
            model.module.load_state_dict(parameters["model"])
        else:
            model.load_state_dict(parameters["model"])
        if mode == "train":
            trained_epoch = parameters["trained_epoch"]
            if "adamw" == parameters["optimizer_name"]:
                optimizer.load_state_dict(parameters["optimizer"])
            else:
                logger.warning("Optimizer changed, do not load parameters of optimizer.")

            if "global_step" in parameters:
                global_step = parameters["global_step"]
    except Exception as e:
        information = "Cannot load checkpoint file with error %s" % str(e)
        if mode == "test":
            logger.error(information)
            raise e
        else:
            logger.warning(information)

    result["model"] = model
    if mode == "train":
        result["optimizer"] = optimizer
        result["trained_epoch"] = trained_epoch
        # result["output_function"] = output_function
        result["global_step"] = global_step

    logger.info("Initialize done.")

    return result

def output_function(data, *args, **params):
    if data['pre_num'] != 0 and data['actual_num'] != 0:
        pre = data['right'] / data['pre_num']
        recall = data['right'] / data['actual_num']
        if pre + recall == 0:
            f1 = 0
        else:
            f1 = 2 * pre * recall / (pre + recall)
    else:
        pre = 0
        recall = 0
        f1 = 0

    metric = {
            'precision': round(pre, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
        }
    return json.dumps(metric)