from fitness import *
from tqdm import tqdm
import torch
from NSGA2 import run_NSGA2
from POBA_GA import run_POBA_GA

def evaluate(model, dataloader, poba = False):
    """
    model, dataloader, MOEA_algorithm이 input으로 들어올 때, 
    주어진 모델과 dataloader에 있는 모든 이미지에 대해 MOEA_algorithm을 수행한 후,
    해당 데이터 셋에 대한 평균 attack success rate, perturbation, query_cnt를 구한다.
    
    """    
    attack_success_rate = 0.
    perturbation = {
        'L0' : 0.,
        'L1' : 0.,
        'L2' : 0.,
        'L_inf' : 0.,
        'Z' : 0.,
        'Z_attack' : 0
    }
    query_cnt = 0
    
    ####################################################
    ####################################################
    ################ implement in here #################
    ####################################################
    ####################################################

    fitness_fn = (attack_fitness, perturbation_fitness)
    
    #Adapted from POBA-GA paper
    pop_size = 50
    n_generation = 150

    cnt = 0
    
    for data in tqdm(dataloader):
        z_status.reset()
        if poba:
            result = run_POBA_GA(model, data[0][0], pop_size, n_generation, fitness_fn, checkpoints = [n_generation])
            result_fit, result_img = result[0]
            cnt+=1
            a, b, c = result_fit[0]
            #print(a)
            true_label = data[1][0]
            # print(c,true_label)
            if (c != true_label):
                attack_success_rate += 1
            perturbation['Z_attack'] = perturbation['Z_attack'] + b
            perturbation['L0'] = perturbation['L0'] + L0_fitness(result_img[0][0], data[0][0])
            perturbation['L1'] = perturbation['L1'] + L1_fitness(result_img[0][0], data[0][0])
            perturbation['L2'] = perturbation['L2'] + L2_fitness(result_img[0][0], data[0][0])
            perturbation['L_inf'] = perturbation['L_inf'] + Linf_fitness(result_img[0][0], data[0][0])
            perturbation['Z'] = perturbation['Z'] + perturbation_fitness(torch.from_numpy(result_img[0][0]), data[0][0])
        else:
            result = run_NSGA2(model, data[0][0], pop_size, n_generation, fitness_fn, checkpoints = [n_generation])
            result_fit, result_img = result[0]
            for res in result_fit[0]:
                cnt+=1
                a, b, c = res
                #print(a)
                true_label = data[1][0]
                
                # print(c,true_label)
                if (c != true_label):
                    attack_success_rate += 1
                perturbation['Z_attack'] = perturbation['Z_attack'] + b
                perturbation['L0'] = perturbation['L0'] + L0_fitness(result_img[0][0], data[0][0])
                perturbation['L1'] = perturbation['L1'] + L1_fitness(result_img[0][0], data[0][0])
                perturbation['L2'] = perturbation['L2'] + L2_fitness(result_img[0][0], data[0][0])
                perturbation['L_inf'] = perturbation['L_inf'] + Linf_fitness(result_img[0][0], data[0][0])
                perturbation['Z'] = perturbation['Z'] + perturbation_fitness(torch.from_numpy(result_img[0][0]), data[0][0])
    attack_success_rate/=cnt
    perturbation['Z_attack']/=cnt
    perturbation['Z']/=cnt
    perturbation['L0']/=cnt
    perturbation['L1']/=cnt
    perturbation['L2']/=cnt
    perturbation['L_inf']/=cnt

    return attack_success_rate, perturbation, query_cnt

if __name__ == '__main__':
    from models import load_model
    from dataloader import load_dataset, load_dataloader 
    from NSGA2 import run_NSGA2
    
    model_names = ['vgg16']
    dataset_names = ['cifar10','imagenet'] # skip mnist
    attack_methods = ['POBA','NSGA2']
    
    for attack_method in attack_methods:
        for dataset_name in dataset_names:
            for model_name in model_names:
                model = load_model(model_name, dataset_name)
                dataset = load_dataset(dataset_name)
                dataloader = load_dataloader(dataset)
                result = evaluate(model = model, dataloader = dataloader, poba = (attack_method=='POBA'))
                attack_success_rate, perturbation, query_cnt = result
                print("dataset: {} model_name: {} attack_method: {}".format(dataset_name, model_name, attack_method))
                print(attack_success_rate)
                print(perturbation)
                print()