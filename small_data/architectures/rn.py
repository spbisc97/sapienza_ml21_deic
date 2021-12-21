from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck
from torch import nn


class ResNet(ResNet):
    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)
    
    @staticmethod
    def get_classifiers():
        return ['rn18', 'rn34', 'rn50', 'rn101', 'rn152']
    
    @classmethod
    def build_classifier(cls, arch: str, num_classes: int, input_channels: int):
        _, depth = arch.split('rn')
        
        RESNET_CONFIG = {
            18  : { 'block' : BasicBlock, 'layers' : [2, 2, 2, 2] },
            34  : { 'block' : BasicBlock, 'layers' : [3, 4, 6, 3] },
            50  : { 'block' : Bottleneck, 'layers' : [3, 4, 6, 3] },
            101 : { 'block' : Bottleneck, 'layers' : [3, 4, 23, 3] },
            152 : { 'block' : Bottleneck, 'layers' : [3, 8, 36, 3] },
        }
                
        cls_instance = cls(**RESNET_CONFIG[int(depth)], num_classes=num_classes)
        
        if input_channels != 3:
            cls_instance.conv1 = nn.Conv2d(input_channels, cls_instance.conv1.out_channels,
                                           kernel_size=7, stride=2, padding=3, bias=False)
        return cls_instance
    