from mmcv.utils import Registry, build_from_cfg
from torch import nn
from mmcv.cnn import MODELS as MMCV_MODELS
MODELS = Registry('models', parent=MMCV_MODELS)
# BACKBONES = Registry('backbone')
# NECKS = Registry('neck')
# HEADS = Registry('head')
# LOSSES = Registry('loss')
# SEGMENTORS = Registry('segmentor')
BACKBONES = MODELS
NECKS =MODELS
HEADS = MODELS
LOSSES = MODELS
SEGMENTORS =MODELS

def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """

    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg )


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg )


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg )


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    return SEGMENTORS.build(cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
