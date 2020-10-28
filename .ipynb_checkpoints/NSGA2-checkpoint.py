def initialize_population(pop_size):
    pass

def fast_non_dominated_sort():
    pass

def crowding_distance():
    pass

def crossover():
    pass

def mutation():
    pass

def run_NSGA2(image, pop_size, n_generation,fitness_fn):
    """
    하나의 이미지가 들어왔을 때 NSGA2 알고리즘을 활용하여 2개의 fitness function에 대한 pareto-front를 구한다.
    위의 initialize_population, fast_non_dominated_sort, crowding_distance, crossover, mutation 등의 논문의 함수를 구현하여 사용한다.
    
    최종 아웃풋 형태 : pareto_front = [(f1,f2),(f1,f2), ... ] : list of tuples 
    """
    fitness_fn_1,fitness_fn_2 = fitness_fn
    pareto_front = []

    ####################################################
    ####################################################
    ################ implement in here #################
    ####################################################
    ####################################################
    
    return pareto_front

def visualize(pareto_front, path):
    """
    pareto front를 이미지 파일로 저장한다.
    """
    
    ####################################################
    ####################################################
    ################ implement in here #################
    ####################################################
    ####################################################    
    
    return 

if __name__ == '__main__':
    from fitness import attack_fitness,perturbation_fitness
    import cv2
    
    pareto_front = run_NSGA2(image, pop_size = 10, n_generation = 50, fitness_fn = (attack_fitness,perturbation_fitness))