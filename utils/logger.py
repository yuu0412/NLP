import logging
import sys

def init_logger(name):  
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler1 = logging.FileHandler(f'logs/{name}.log')
    logger.addHandler(handler1)
    handler2 = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler2)

    return logger