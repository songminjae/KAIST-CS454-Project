"""
load datasets, dataloader
"""
import numpy as np
import cv2
import torch
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, ImageNet
from torch.utils.data import DataLoader

def get_preprocess(size,mean,std):
    mean = np.array(mean)[...,np.newaxis,np.newaxis]
    std = np.array(std)[...,np.newaxis,np.newaxis]
    def preprocess(x):
        # W,H,3 -> 3,W,H
        x = np.rollaxis(cv2.resize(x,(size,size)),-1,0)
        x = (x - mean)/std
        x = torch.Tensor(x)
        return x
    return preprocess

def load_dataset(dataset_name):
    def _transform(pil):
        return np.array(pil).astype('float32')/255.
    if dataset_name == 'mnist':
        dataset = MNIST(root='./datasets/', train = False,transform = _transform, download=True)
    if dataset_name == 'cifar10':
        dataset = CIFAR10(root='./datasets/', train = False,transform = _transform, download=True)
    if dataset_name == 'imagenet':
        dataset = ImageNet(root='./datasets/', train = False,transform = _transform, download=True)
    return dataset

def load_dataloader(dataset):
    """
    dataloader batchsize == 1
    return W,H,3 numpy arr
    """
    def _collate_fn(datas):
        xs = []
        ys = []
        for x,y in datas:
            xs.append(x)
            ys.append(y)
        return xs,ys
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False,collate_fn= _collate_fn)
    return dataloader

if __name__ =='__main__':
    import torch.nn.functional as F
    from models import load_model
    
    def give_perturbation(x):
        return x
    
    use_cuda = True
    dataset = load_dataset('cifar10')
    dataloader = load_dataloader(dataset)
    preprocess = get_preprocess(size=32,mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])
    model = load_model('vgg16','cifar10',use_cuda = use_cuda)
    model.eval()
    """
    example usage for evaluation.py
    """
    for idx,(x,y) in enumerate(dataloader):
        x = [give_perturbation(img) for img in x]
        x = [preprocess(img) for img in x] ## img : 3,W,H
        x = torch.stack(x,0)

        if use_cuda:
            x = x.cuda()

        with torch.no_grad():
            logits = model(x)
            prediction = F.softmax(logits,dim = -1)
            prediction = prediction.cpu().detach().numpy()
        
        print("prediction",prediction[0])
        print("true_label",y[0])
        
        break
    
    """
    example usage for NSGA2.py
    """
    img = np.zeros((32,32,3)).astype('float32') ## 32,32,3 ndarray
    
    img = give_perturbation(img)
    img = preprocess(img)
    img = torch.Tensor(img).unsqueeze(0)
    img = img.cuda()
    with torch.no_grad():
        logits = model(img)
        prediction = F.softmax(logits,dim = -1)
        prediction = prediction.cpu().detach().numpy()
    
    print("prediction",prediction[0])