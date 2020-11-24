import torchattacks
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import load_model
from dataloader import *
from tqdm import tqdm

def get_attack(attack_method,model,eps = 8/255):
    if attack_method == 'fgsm':
        return torchattacks.FGSM(model, eps = eps)
    if attack_method == 'deepfool':
        return torchattacks.DeepFool(model, steps = 3)
    if attack_method == 'bim':
        return torchattacks.BIM(model, eps = eps, alpha = eps/4, steps = 7)        
    if attack_method == 'cw':
        return torchattacks.CW(model, c=1, kappa = 0, steps = 1000, lr = 0.01)
    if attack_method == 'rfgsm':
        return torchattacks.RFGSM(model, eps = eps, alpha = eps/2, steps = 1)
    if attack_method == 'pgd':
        return torchattacks.PGD(model, eps = eps, alpha = eps/4, steps = 7)
    if attack_method == 'ffgsm':
        return torchattacks.FFGSM(model, eps = eps, alpha = 1.5*eps)
    if attack_method == 'tpgd':
        return torchattacks.TPGD(model, eps = eps, alpha = eps/4, steps = 7)
    if attack_method == 'mifgsm':
        return torchattacks.MIFGSM(model, eps = eps, decay = 0.9, steps = 5)
    if attack_method == 'multiattack':
        return torchattacks.MultiAttack(model,
                                        [torchattacks.PGD(model, eps=eps, alpha=eps/4, steps=7, random_start=True)]*10)             

def evaluate(model, attack_method, dataloader, eps, use_cuda = True):
    attack_success_rate = 0.
    perturbation = {
        'L0' : 0.,
        'L1' : 0.,
        'L2' : 0.,
        'L_inf' : 0.,
        'Z' : 0.
    }
    
    atk = get_attack(attack_method = attack_method, model = model, eps = eps)
    ys = []
    for idx,(x,y) in enumerate(tqdm(dataloader)):
        x = [preprocess(img) for img in x]
        x = torch.stack(x,0)
        y = torch.Tensor(y).type(torch.LongTensor)
        if use_cuda:
            x = x.cuda()
            y = y.cuda()
        adversarial_x = atk(x,y)
        with torch.no_grad():
            adversarial_y = F.softmax(model(adversarial_x))
            adversarial_y = torch.argmax(adversarial_y,dim = -1)
        
        for a,b in zip(list(adversarial_y.data),list(y.data)):
            if a!=b:
                attack_success_rate += 1/x.shape[0]
        
        l0 = torch.norm(x - adversarial_x, p = 0)/(x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3])
        l1 = torch.norm(x - adversarial_x, p = 1)/(x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3])
        l2 = torch.norm(x - adversarial_x, p = 2)/(x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3])
        l_inf = torch.norm(x - adversarial_x, p = float('inf'))
        
        z = 0
        
        perturbation['L0'] += l0.item()/len(dataloader)
        perturbation['L1'] += l1.item()/len(dataloader)
        perturbation['L2'] += l2.item()/len(dataloader)
        perturbation['L_inf'] += l_inf.item()/len(dataloader)
        perturbation['Z'] += z / len(dataloader)
        
    attack_success_rate /= len(dataloader)
        
    return attack_success_rate, perturbation

#adversarial_images = atk(images, labels)

if __name__ == '__main__':
    batch_size = 128
    
    use_cuda = True
    #attack_methods = ['fgsm','deepfool','bim','cw','rfgsm','pgd','ffgsm','tpgd','mifgsm','multiattack']
    attack_methods = ['fgsm']
    model_names = ['vgg16','resnet50','inception_v3']
    dataset_names = ['imagenet','cifar10'] # skip mnist
    epss = [1/255]
    #epss = [1/255, 2/255,4/255,8/255,16/255,32/255]
    #epss = [8/255]
    
    for dataset_name in dataset_names:
        for model_name in model_names:
            for attack_method in attack_methods:
                for eps in epss:
                    
                    dataset = load_dataset(dataset_name)
                    dataloader = load_dataloader(dataset, batch_size)
                    if dataset_name == 'cifar10':
                        preprocess = get_preprocess(size = 32)
                    if dataset_name == 'imagenet':
                        if model_name == 'inception_v3':
                            preprocess = get_preprocess(size = 299)
                        else:
                            preprocess = get_preprocess(size = 224)
                    model = load_model(model_name,dataset_name,use_cuda = use_cuda)
                    model.eval()

                    attack_success_rate, perturbation = evaluate(model,attack_method,dataloader,eps)
                    
                    print(dataset_name, model_name, attack_method, eps)
                    print(attack_success_rate,perturbation)