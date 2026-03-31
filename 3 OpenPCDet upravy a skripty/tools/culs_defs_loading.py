import pcdet
import pcdet.config, pcdet.datasets, pcdet.models, pcdet.utils
import logging
from easydict import EasyDict
from pathlib import Path
from collections import namedtuple
import train_utils.optimization as opt_utils
import torch, random, numpy

Dataset = namedtuple('Dataset', 'set loader sampler')

def ensure_determinism(seed=12345):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.use_deterministic_algorithms(True)

def load_trial_params(study, cfg):
    params = study.best_trial.params
    layer_nums, strides, num_filters, num_upsample_filters = [], [], [], []
    for i in range(params["2d_length"]):
        layer_nums.append(params[f"2d_layer_num{i}"])
        num_filters.append(params[f"2d_filter{i}"])
        num_upsample_filters.append(params[f"2d_upsample_filter{i}"])
        strides.append(1)
    strides[-1] = params["2d_last_stride"]
    cfg.MODEL.BACKBONE_2D.LAYER_NUMS = layer_nums
    cfg.MODEL.BACKBONE_2D.NUM_FILTERS = num_filters
    cfg.MODEL.BACKBONE_2D.NUM_UPSAMPLE_FILTERS = num_upsample_filters
    cfg.MODEL.BACKBONE_2D.LAYER_STRIDES = strides
    cfg.MODEL.BACKBONE_2D.UPSAMPLE_STRIDES = strides
    
    anchors = []
    for i in range(params["dense_anchor_len"]):
        anchors.append([
            2**params[f"dense_anchor{i}_x"],
            2**params[f"dense_anchor{i}_y"],
            2**params[f"dense_anchor{i}_z"],
        ])
    cfg.MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG[0].anchor_sizes = anchors
    
    cfg.OPTIMIZATION.LR = params["opt_lr"]
    cfg.OPTIMIZATION.WEIGHT_DECAY = params["opt_weight_decay"]
    cfg.OPTIMIZATION.MOMENTUM = params["opt_momentum"]
    # must be 2 values here!
    cfg.OPTIMIZATION.MOMS = [
        params["opt_moms_0"],
        params["opt_moms_1"]
    ]
    cfg.OPTIMIZATION.PCT_START = params["opt_pct_start"]
    cfg.OPTIMIZATION.DIV_FACTOR = params["opt_div_factor"]
    cfg.OPTIMIZATION.LR_DECAY = params["opt_lr_decay"]
    cfg.OPTIMIZATION.WARMUP_EPOCH = params["opt_warmup"]
    cfg.OPTIMIZATION.LR_WARMUP = not (cfg.OPTIMIZATION.WARMUP_EPOCH == 0)
    cfg.OPTIMIZATION.GRAD_NORM_CLIP = params["opt_grad_norm_clip"]
    
def load_config(cfg_file, epochs=None):
    cfg = pcdet.config.cfg
    pcdet.config.cfg_from_yaml_file(cfg_file, cfg)
    cfg.TAG = Path(cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(cfg_file.split('/')[1:-1])
    if epochs is not None:
        cfg.OPTIMIZATION.NUM_EPOCHS = epochs
    return cfg


def create_loaders(cfg, logger, split_name, batch_size=1, workers=1, epochs=0, no_shuffle=False):
    dataset, loader, sampler = pcdet.datasets.build_dataloader_culs(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=batch_size,
        dist=False,
        workers=workers,
        logger=logger,
        training=(split_name in ["train", "val"]) and not no_shuffle,
        total_epochs=epochs,
        seed=2345212,
        split=split_name
    )
    return Dataset(dataset, loader, sampler)


def build_model(cfg, train_data):
    model = pcdet.models.build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=train_data.set
    )
    model.cuda()
    optimizer = opt_utils.build_optimizer(
        model=model,
        optim_cfg=cfg.OPTIMIZATION
    )
    model.train()
    lr_scheduler, lr_warmup_scheduler = opt_utils.build_scheduler(
        optimizer,
        total_iters_each_epoch=len(train_data.loader),
        total_epochs=cfg.OPTIMIZATION.NUM_EPOCHS,
        last_epoch=-1,
        optim_cfg=cfg.OPTIMIZATION
    )
    return model, optimizer, lr_scheduler, lr_warmup_scheduler


# LOGGING

"""
Creates an empty logger and a message formatter. Allows running a script multiple
times inside an IDE without duplicate loggers appearing.
"""

def create_default_logger(level, name=None):
    # A little bit more complicated to prevent duplicate logging when running
    # the script multiple times inside an IDE that keeps the state.
    # if not name:
    #     name = __name__
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    logging.addLevelName(logging.WARNING, "WARN")
    logging.addLevelName(logging.ERROR, "ERR")
    logging.addLevelName(logging.CRITICAL, "CRIT")
    formatter = logging.Formatter('%(asctime)s  %(levelname)4s  %(message)s', "%y-%m-%d %H:%M:%S")
    return logger, formatter
    
def get_console_handle(formatter):
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    return console

"""
Creates a logger that logs into the specified output path and also into console.
Used for normal training.
"""
def create_training_logger(output_path, filename="train.log", console_only=False, level=logging.INFO, name=None):
    log_path = Path(output_path) / (filename)
    logger, formatter = create_default_logger(level, name)
    console = get_console_handle(formatter)
    logger.addHandler(console)
    if log_path is None or console_only:
        return logger
    file_handler = logging.FileHandler(filename=log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # logger.info(f"Output folder created at {output_path}.")
    return logger

"""
Creates a dummy logger that doesn't log anything. This is to give functions that
use a logger have an object to send reports to.
"""
def create_dummy_logger(name=None):
    logger, _ = create_default_logger(logging.INFO, name)
    logger.addHandler(logging.NullHandler())
    return logger

"""
A simple console logger.
"""
def create_console_logger(level=logging.INFO, name=None):
    logger, formatter = create_default_logger(level, name)
    logger.addHandler(get_console_handle(formatter))
    return logger
