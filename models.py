"""
load pretrained models
"""

def load_model(model_name, use_cuda = True):
    """
    load pretrained DNN models
    
    if you want to use GPU, set use_cuda = True
    else,                   set use_cuda = False
    
    """
    
    if model_name == 'vgg16':
        model = None
    elif model_name == 'resnet50':
        model = None
    elif model_name == 'inception_v3':
        model = None
    
    if use_cuda:
        model = model.cuda()
    return model