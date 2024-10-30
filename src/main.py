import argparse
import os
import torch
import logging

from tools.initial import init_all
from tools.train import train
import numpy as np
import random

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    
    set_seed(seed=2024)

    cuda = torch.cuda.is_available()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    logger.info("CUDA available: %s" % str(cuda))

    model_path = "/home/fcy/SCR/Pretrain_model/bert_base_chinese"

    parameters = init_all(model_path, "train")

    do_test = False

    # print(args.comment)
    with torch.autograd.set_detect_anomaly(True):
        train(parameters, do_test)
