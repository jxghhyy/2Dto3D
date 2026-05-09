import warnings

# ---- mmcv → 纯 PyTorch 替换 (兼容 mmcv 2.x / 无 mmcv 环境) ----
from collections import OrderedDict

class _Registry:
    """轻量 Registry 替代，接口兼容 mmcv.utils.Registry 的 build 流程"""
    def __init__(self, name):
        self._name = name
        self._module_dict = OrderedDict()

    def __repr__(self):
        return f'Registry({self._name}, keys={list(self._module_dict.keys())})'

    def register_module(self, name=None, force=False, module=None):
        """Decorator / functional 用法兼容"""
        if module is not None:
            # functional: register_module(module=SomeClass)
            n = name or module.__name__
            self._module_dict[n] = module
            return module
        # decorator: @register_module()
        def _register(cls):
            n = name or cls.__name__
            self._module_dict[n] = cls
            return cls
        return _register

def build_from_cfg(cfg, registry, default_args=None):
    """从 cfg dict 构建模块，兼容 mmcv.utils.build_from_cfg"""
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, got {type(cfg)}')
    _cfg = cfg.copy()
    _type = _cfg.pop('type')
    cls = registry._module_dict.get(_type)
    if cls is None:
        raise KeyError(f'{_type} not found in {registry}')
    if default_args is not None:
        for k, v in default_args.items():
            _cfg.setdefault(k, v)
    return cls(**_cfg)

BACKBONES = _Registry('backbone')
NECKS = _Registry('neck')
HEADS = _Registry('head')
LOSSES = _Registry('loss')
SEGMENTORS = _Registry('segmentor')


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
    return build(cfg, BACKBONES)


def build_neck(cfg):
    """Build neck."""
    return build(cfg, NECKS)


def build_head(cfg):
    """Build head."""
    return build(cfg, HEADS)


def build_loss(cfg):
    """Build loss."""
    return build(cfg, LOSSES)


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
