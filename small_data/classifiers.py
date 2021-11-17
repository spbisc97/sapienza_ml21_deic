import torch
from torch import nn, Tensor
import torchvision

from .architectures import cifar_resnet, wrn
from functools import partial


CLASSIFIERS = [
    'rn18', 'rn34', 'rn50', 'rn101', 'rn152',  # Standard ResNet
    'rn20', 'rn32', 'rn44', 'rn56', 'rn110', 'rn1202',  # CIFAR ResNet
    'wrn-16-8', 'wrn-16-10', 'wrn-22-8', 'wrn-22-10', 'wrn-28-10', 'wrn-28-12',  # Wide ResNet
]


def available_classifiers():
    """ Returns a list of all architectures supported by `build_classifier`. """

    return list(CLASSIFIERS)


def build_classifier(arch: str, num_classes: int, input_channels: int = 3) -> nn.Module:
    """ Instantiates a given neural network architecture.

    Parameters
    ----------
    arch : str
        The name of the model architecture.
        A list of supported architectures can be obtained from `available_classifiers`.
    num_classes : int
        The number of classes to be distinguished, i.e., the number of output neurons.
    input_channels : int, default: 3
        The number of input channels.
    
    Returns
    -------
    torch.nn.Module
        The model with random initialization.
    """

    if arch not in CLASSIFIERS:
        raise ValueError('Unknown classifier: ' + arch)
    
    func_name = f'build_{arch.replace("-", "_")}_classifier'
    return globals()[func_name](num_classes, input_channels)


### ResNet ###

RESNET_CONFIG = {
    18  : { 'block' : torchvision.models.resnet.BasicBlock, 'layers' : [2, 2, 2, 2] },
    34  : { 'block' : torchvision.models.resnet.BasicBlock, 'layers' : [3, 4, 6, 3] },
    50  : { 'block' : torchvision.models.resnet.Bottleneck, 'layers' : [3, 4, 6, 3] },
    101 : { 'block' : torchvision.models.resnet.Bottleneck, 'layers' : [3, 4, 23, 3] },
    152 : { 'block' : torchvision.models.resnet.Bottleneck, 'layers' : [3, 8, 36, 3] },
}

def build_rn_classifier(num_layers: int, num_classes: int, input_channels: int = 3) -> nn.Module:
    """ Creates a ResNet model.

    Parameters
    ----------
    num_layers : {18, 34, 50, 101, 152}
        Depth of the network.
    num_classes : int
        The number of classes to be distinguished, i.e., the number of output neurons.
    input_channels : int, default: 3
        The number of input channels.
    
    Returns
    -------
    torch.nn.Module
        ResNet module with random initialization.
    """

    model = torchvision.models.resnet.ResNet(num_classes=num_classes, **RESNET_CONFIG[num_layers])
    if input_channels != 3:
        model.conv1 = nn.Conv2d(input_channels, model.conv1.out_channels,
                                kernel_size=7, stride=2, padding=3, bias=False)
    return model

build_rn18_classifier = partial(build_rn_classifier, 18)
build_rn34_classifier = partial(build_rn_classifier, 34)
build_rn50_classifier = partial(build_rn_classifier, 50)
build_rn101_classifier = partial(build_rn_classifier, 101)
build_rn152_classifier = partial(build_rn_classifier, 152)


### CIFAR ResNet ###

CIFAR_RESNET_CONFIG = {
    20    : { 'block' : cifar_resnet.BasicBlock, 'layers' : [3, 3, 3] },
    32    : { 'block' : cifar_resnet.BasicBlock, 'layers' : [5, 5, 5] },
    44    : { 'block' : cifar_resnet.BasicBlock, 'layers' : [7, 7, 7] },
    56    : { 'block' : cifar_resnet.BasicBlock, 'layers' : [9, 9, 9] },
    101   : { 'block' : cifar_resnet.BasicBlock, 'layers' : [18, 18, 18] },
    1202  : { 'block' : cifar_resnet.BasicBlock, 'layers' : [200, 200, 200] },
}

def build_cifar_rn_classifier(num_layers: int, num_classes: int, input_channels: int = 3) -> nn.Module:

    return cifar_resnet.ResNet(num_classes=num_classes, input_channels=input_channels, **CIFAR_RESNET_CONFIG[num_layers])

build_rn20_classifier = partial(build_cifar_rn_classifier, 20)
build_rn32_classifier = partial(build_cifar_rn_classifier, 32)
build_rn44_classifier = partial(build_cifar_rn_classifier, 44)
build_rn56_classifier = partial(build_cifar_rn_classifier, 56)
build_rn110_classifier = partial(build_cifar_rn_classifier, 101)
build_rn1202_classifier = partial(build_cifar_rn_classifier, 1202)


### Wide ResNet ###

def build_wrn_classifier(num_layers: int, widen_factor: int, num_classes: int, input_channels: int = 3) -> nn.Module:
    return wrn.WideResNet(num_layers, num_classes, input_channels=input_channels, widen_factor=widen_factor)

build_wrn_16_8_classifier = partial(build_wrn_classifier, 16, 8)
build_wrn_16_10_classifier = partial(build_wrn_classifier, 16, 10)
build_wrn_22_8_classifier = partial(build_wrn_classifier, 22, 8)
build_wrn_22_10_classifier = partial(build_wrn_classifier, 22, 10)
build_wrn_28_10_classifier = partial(build_wrn_classifier, 28, 10)
build_wrn_28_12_classifier = partial(build_wrn_classifier, 28, 12)
