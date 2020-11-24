"""
load pretrained models
"""
import torchvision.models as models
import torch.nn as nn
import torch
from PyTorch_CIFAR10 import cifar10_models

class Normalize(nn.Module):
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std

def load_model(model_name, dataset_name ,use_cuda = True):
    """
    load pretrained DNN models
    
    if you want to use GPU, set use_cuda = True
    else,                   set use_cuda = False
    
    """
    
    assert model_name in ['vgg16','resnet50','inception_v3']
    assert dataset_name in ['imagenet','cifar10']
    
    if dataset_name == 'imagenet':
        if model_name == 'vgg16':
            model = models.vgg16_bn(pretrained=True) ## 3,224,224
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True) ## 3,224,224
        elif model_name == 'inception_v3':
            model = models.inception_v3(pretrained=True) ## 3,299,299
    elif dataset_name == 'cifar10':
        if model_name == 'vgg16':
            model = cifar10_models.vgg16_bn(pretrained=True) ## 
        elif model_name == 'resnet50':
            model = cifar10_models.resnet50(pretrained=True) ## 
        elif model_name == 'inception_v3':
            model = cifar10_models.inception_v3(pretrained=True) ##         
    elif dataset_name == 'mnist':
        raise Exception('Not Implemented')
    
    model = nn.Sequential(Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                          model)
    
    if use_cuda:
        model = model.cuda()
    model.eval()
    
    return model

if __name__ == '__main__':
    model_names = ['vgg16','resnet50','inception_v3']
    dataset_names = ['imagenet','cifar10']
    for model_name,dataset_name in zip(model_names,dataset_names):
        model = load_model(model_name,dataset_name)