import torchattacks
import torch
import torch.nn as nn
import torch.nn.functional as F
from blackbox_attack.zoo_attack import zoo_attack
from models import load_model
from dataloader import *
from tqdm import tqdm

def get_attack(attack_method,model,n_classes = 10,eps = 8/255,batch_size = 128):
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
    if attack_method == 'zoo':
        return zoo_attack(model,n_classes,batch_size)
    
def evaluate(model, attack_method, dataloader, eps,n_classes = 10, batch_size = 16, use_cuda = True):
    attack_success_rate = 0.
    perturbation = {
        'L0' : 0.,
        'L1' : 0.,
        'L2' : 0.,
        'L_inf' : 0.,
        'Z' : 0.
    }
    
    atk = get_attack(attack_method = attack_method, model = model, n_classes = n_classes, eps = eps, batch_size = batch_size)
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
        
        ## not implemented yet
        z = 0
        
        perturbation['L0'] += l0.item()/len(dataloader)
        perturbation['L1'] += l1.item()/len(dataloader)
        perturbation['L2'] += l2.item()/len(dataloader)
        perturbation['L_inf'] += l_inf.item()/len(dataloader)
        perturbation['Z'] += z / len(dataloader)
        
        #print(attack_success_rate)
        #print(l0.item(),l1.item(),l2.item(),l_inf.item())
        
    attack_success_rate /= len(dataloader)
        
    return attack_success_rate, perturbation


if __name__ == '__main__':
    batch_size = 64
    
    use_cuda = True
    attack_methods = ['fgsm','deepfool','bim','cw','rfgsm','pgd','ffgsm','tpgd','mifgsm','multiattack','zoo']
    #attack_methods = ['zoo']
    model_names = ['vgg16','resnet50','inception_v3']
    dataset_names = ['cifar10','imagenet'] # skip mnist
    epss = [1/255, 2/255,4/255,8/255,16/255]
    
    for dataset_name in dataset_names:
        for model_name in model_names:
            for attack_method in attack_methods:
                for eps in epss:
                    dataset = load_dataset(dataset_name)
                    dataloader = load_dataloader(dataset, batch_size)
                    if dataset_name == 'cifar10':
                        preprocess = get_preprocess(size = 32)
                        n_classes = 10
                    if dataset_name == 'imagenet':
                        if model_name == 'inception_v3':
                            preprocess = get_preprocess(size = 299)
                        else:
                            preprocess = get_preprocess(size = 224)
                        n_classes = 1000
                    model = load_model(model_name,dataset_name,use_cuda = use_cuda)
                    model.eval()

                    attack_success_rate, perturbation = evaluate(model,attack_method,dataloader,eps,n_classes)
                    
                    print("dataset: {} model_name: {} attack_method: {} eps: {}".format(dataset_name, model_name, attack_method, eps))
                    print(attack_success_rate,perturbation)
                    print()