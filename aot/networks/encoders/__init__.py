from SAMTrack.aot.networks.encoders.mobilenetv2 import MobileNetV2
from SAMTrack.aot.networks.encoders.mobilenetv3 import MobileNetV3Large
from SAMTrack.aot.networks.encoders.resnet import ResNet101, ResNet50
from SAMTrack.aot.networks.encoders.resnest import resnest
from SAMTrack.aot.networks.encoders.swin import build_swin_model
from SAMTrack.aot.networks.layers.normalization import FrozenBatchNorm2d
from torch import nn


def build_encoder(name, frozen_bn=True, freeze_at=-1):
    if frozen_bn:
        BatchNorm = FrozenBatchNorm2d
    else:
        BatchNorm = nn.BatchNorm2d

    if name == 'mobilenetv2':
        return MobileNetV2(16, BatchNorm, freeze_at=freeze_at)
    elif name == 'mobilenetv3':
        return MobileNetV3Large(16, BatchNorm, freeze_at=freeze_at)
    elif name == 'resnet50':
        return ResNet50(16, BatchNorm, freeze_at=freeze_at)
    elif name == 'resnet101':
        return ResNet101(16, BatchNorm, freeze_at=freeze_at)
    elif name == 'resnest50':
        return resnest.resnest50(norm_layer=BatchNorm,
                                 dilation=2,
                                 freeze_at=freeze_at)
    elif name == 'resnest101':
        return resnest.resnest101(norm_layer=BatchNorm,
                                  dilation=2,
                                  freeze_at=freeze_at)
    elif 'swin' in name:
        return build_swin_model(name, freeze_at=freeze_at)
    else:
        raise NotImplementedError
