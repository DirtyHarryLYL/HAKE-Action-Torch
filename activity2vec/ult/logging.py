# Copied from slowfast/utils/logging.py of SlowFast repository: 
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/utils/logging.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import os
import sys
from easydict import EasyDict as edict

def setup_logging(output_dir=None, func='train'):
    """
    Sets up the logging.
    """
    assert func in ['train', 'test', 'inference']

    # Set up logging format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO, format=_FORMAT, stream=sys.stdout
    )

    loggers = edict()

    if func == 'train':
        modes = ['train', 'test']
    elif func == 'test':
        modes = ['test']
    else:
        modes = ['Activity2Vec']
    
    for mode in modes:
        logger = logging.getLogger(mode)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        plain_formatter = logging.Formatter(
            # "[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
            "[%(asctime)s][%(levelname)s] %(name)s: %(message)s",
            datefmt="%m/%d %H:%M:%S",
        )

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(plain_formatter)
        logger.addHandler(ch)

        if output_dir is not None:
            filename = os.path.join(output_dir, "{}.log".format(mode))
            fh = logging.StreamHandler(open(filename, "a"))
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(plain_formatter)
            logger.addHandler(fh)

        loggers[mode] = logger

    return loggers