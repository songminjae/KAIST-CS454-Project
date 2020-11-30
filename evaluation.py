query_cnt = 0
from fitness import *
import torch

def evaluate(model, dataloader, MOEA_algorithm):
    """
    model, dataloader, MOEA_algorithm이 input으로 들어올 때, 
    주어진 모델과 dataloader에 있는 모든 이미지에 대해 MOEA_algorithm을 수행한 후,
    해당 데이터 셋에 대한 평균 attack success rate, perturbation, query_cnt를 구한다.
    
    """
    global query_cnt
    
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
    n_generation = 400

    cnt = 0
    
    for data in dataloader:
        z_status.reset()
        result_fit, result_img = MOEA_algorithm(model, data[0][0], pop_size, n_generation, fitness_fn)
        result = result_fit[0]
        for res in result:
            cnt+=1
            a, b = res
            attack_success_rate += a
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
    
    model = load_model('vgg16', 'cifar10')
    dataset = load_dataset('imagenet')
    dataloader = load_dataloader(dataset)
    
    attack_success_rate, perturbation, query_cnt = evaluate(model = model, dataloader = dataloader, MOEA_algorithm = run_NSGA2)
    print(attack_success_rate)
    print(perturbation)
    print(query_cnt)