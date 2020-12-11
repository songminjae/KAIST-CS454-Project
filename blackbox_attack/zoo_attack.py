import time
import random 
import numpy as np
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
alpha = 0.2
beta = 0.001

def coordinate_ADAM(losses, indice, grad,  batch_size, mt_arr, vt_arr, real_modifier, lr, adam_epoch, beta1, beta2):
    for i in range(batch_size):
        grad[i] = (losses[i*2] - losses[i*2+1]) / 0.0002 
    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * grad
    mt_arr[indice] = mt
    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (grad * grad)
    vt_arr[indice] = vt
    # epoch is an array; for each index we can have a different epoch number
    epoch = adam_epoch[indice]
    corr = (np.sqrt(1 - np.power(beta2,epoch))) / (1 - np.power(beta1, epoch))
    m = real_modifier.reshape(-1)
    old_val = m[indice] 
    old_val -= lr * corr * mt / (np.sqrt(vt) + 1e-8)
    m[indice] = old_val
    adam_epoch[indice] = epoch + 1


def attack(input, label, net, n_classes, c = 1, batch_size= 128, TARGETED=False):
    index = label.view(-1,1)
    label_onehot = torch.FloatTensor(input.shape[0] , n_classes).cuda()
    label_onehot.zero_()
    label_onehot.scatter_(1,index,1)
    label_onehot_v = Variable(label_onehot, requires_grad = False)
    var_size = input.view(-1).size()[0]
    real_modifier = torch.FloatTensor(input.size()).zero_().cuda()
    with torch.no_grad():
        for iter in range(300):
            random_set = np.random.permutation(var_size)
            losses = np.zeros(2*batch_size, dtype=np.float32)
            #print(torch.sum(real_modifier))
            for i in range(2*batch_size):
                modifier = real_modifier.clone().view(-1).cuda()
                if i%2==0:
                    modifier[random_set[i//2]] += 0.0001 
                else:
                    modifier[random_set[i//2]] -= 0.0001
                modifier = modifier.view(input.size())
                output = net(torch.clamp(input + modifier,0,1))
                #print(output)
                real = torch.max(torch.mul(output, label_onehot_v), 1)[0]
                other = torch.max(torch.mul(output, (1-label_onehot_v))-label_onehot_v*10000,1)[0]
                loss1 = torch.sum(modifier*modifier)/1
                if TARGETED:
                    loss2 = c* torch.sum(torch.clamp(other - real, min=0))
                else:
                    loss2 = c* torch.sum(torch.clamp(real - other, min=0))
                error = loss2 + loss1 
                losses[i] = error.item()
            #if (iter+1)%1 == 0:
            #    print(np.sum(losses))
            #if loss2.data[0]==0:
            #    break
            grad = np.zeros(batch_size, dtype=np.float32)
            mt = np.zeros(var_size, dtype=np.float32)
            vt = np.zeros(var_size, dtype=np.float32)
            adam_epoch = np.ones(var_size, dtype = np.int32)
            np_modifier = real_modifier.cpu().numpy()
            lr = 0.1
            beta1, beta2 = 0.9, 0.999
            #for i in range(1):
            #print(np.count_nonzero(np_modifier))
            coordinate_ADAM(losses, random_set[:batch_size], grad, batch_size, mt, vt, np_modifier, lr, adam_epoch, beta1, beta2)
            real_modifier = torch.from_numpy(np_modifier)
    #print(torch.norm(real_modifier_v)) 
    return (input + real_modifier.cuda())

def zoo_attack(model,n_classes,batch_size):
    def att(x,y):
        return attack(x,y,model,int(n_classes),c=1, batch_size=batch_size)
    return att