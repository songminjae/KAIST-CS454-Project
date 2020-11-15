import numpy as np
import cv2
import matplotlib.pyplot as plt
# how about number of noise point, nois point size? How?


# initialize perturbations with size of pop_size, with list of random gaussian distribution
def initialize_population(pop_size, origin_img):
    img = np.array(origin_img)
    init_pop = []
    for i in range(pop_size):
        variance = np.random.normal(0, 1) + 1
        noise = np.random.normal(
            0, variance, size=img.shape)  # varies variance when picking noise
        init_pop.append(noise)
    # 나중에 사용할 때, np.clip(img + noise, 0, 1)으로 visible한 이미지 만들 수 있을 듯
    return init_pop


def fast_non_dominated_sort(fitness_fn):
    fronts = [[]]
    S = [[] for i in range(len(fitness_fn))]
    n = [-1 for i in range(len(fitness_fn))]

    for p in range(len(fitness_fn)):
        S[p] = []
        n[p] = 0
        for q in range(len(fitness_fn))[p+1:]:
            if dominate(fitness_fn[p], fitness_fn[q]):  # TODO
                n[q] = n[q] + 1
                S[p].append(q)
            elif dominate(fitness_fn[q], fitness_fn[p]):
                n[p] = n[p] + 1
                S[q].append(p)
        if n[p] == 0:
            fronts[0].append(p)

    cnt = 0
    while (fronts[cnt]):
        next_front = []
        for p in fronts[cnt]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0 and q not in next_front:
                    next_front.append(q)
        fronts.append(next_front)
        cnt = cnt + 1

    return fronts[:len(fronts)-1]


def dominate(f1, f2):
    if (f1[0] > f2[0] and f1[1] > f2[1]) or (f1[0] >= f2[0] and f1[1] > f2[1]) or (f1[0] > f2[0] and f1[1] >= f2[1]):
        return True
    return False


def crowding_distance(fitness):
    distances = [0.0] * len(fitness)
    crowd = [(f_value, i) for i, f_value in enumerate(fitness)]

    for i in range(2):
        crowd.sort(key=lambda element: element[0][i])
        distances[crowd[0][1]] = float("inf")
        distances[crowd[-1][1]] = float("inf")
        normalization = float(crowd[-1][0][i] - crowd[0][0][i])
        if normalization == 0:
            continue
        for j in range(1, len(fitness)-1):
            distances[j] += (crowd[j+1][0][i] - crowd[j-1]
                             [0][i]) / normalization

    return distances


def crossover(prob_c, p1, p2):  # input of 2 different perturbation with np.array
    a1, a2 = np.array(p1), np.array(p2)
    assert a1.shape == a2.shape
    b = np.random.randint(0, 2, a1.shape)
    b_not = np.ones(a1.shape, int) - b
    a1_cross, a2_cross = a1 * b + a2 * b_not, a1 * b_not + a2 * b
    return a1_cross, a2_cross


def mutation(prob_m, p):
    a = np.array(p)
    c = np.ones(a.shape)
    # mutation 이 생기는 pixel은 몇개?? not mentioned in paper...?
    c[np.random.randint(a.shape[0])][np.random.randint(
        a.shape[1])] = np.random.uniform(0, 2)
    if np.random.rand() < prob_m:
        a = a * c
    return a


def run_NSGA2(model, image, pop_size, n_generation, fitness_fn):
    """
    하나의 이미지가 들어왔을 때 NSGA2 알고리즘을 활용하여 2개의 fitness function에 대한 pareto-front를 구한다.
    위의 initialize_population, fast_non_dominated_sort, crowding_distance, crossover, mutation 등의 논문의 함수를 구현하여 사용한다.

    최종 아웃풋 형태 : pareto_front = [[f1,f2], [f1,f2], ...] : list of lists
    """

    ###### TODO (만들어야 할 함수들) ######

    # do_selection_crossover_mutation(parent): input은 부모 이미지 리스트, output은 자식 이미지 리스트
    # 이 때 selection의 우선순위: rank가 낮을수록, crowding distance가 클수록 우선순위 올라감
    # fitness(temp): input은 이미지 리스트, output은 각 이미지의 피트니스 값들(attack success rate & perturbation의 pair 형태)로 이루어진 리스트([[f1, f2], [f1, f2], ...]])

    fitness_fn_1, fitness_fn_2 = fitness_fn
    pareto_front = []

    iteration = 0
    parent = initialize_population(pop_size)
    offspring = do_selection_crossover_mutation(parent)  # TODO

    while (iteration < n_generation):
        temp = parent + offspring
        pareto_front = fast_non_dominated_sort(fitness(temp))  # TODO
        parent = []
        cnt = 0
        crowding_distance_list = []
        while (len(parent) + len(pareto_front) < pop_size):
            crowding_distance_list.append(crowding_distance(pareto_front))
            parent = parent + pareto_front[cnt]
            cnt = cnt + 1
        parent = parent + pareto_front[:(pop_size - len(parent))]
        offspring = do_selection_crossover_mutation(parent)
        iteration = iteration + 1

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
    from fitness import attack_fitness, perturbation_fitness
    import cv2

    pareto_front = run_NSGA2(model, image, pop_size=10, n_generation=50, fitness_fn=(
        attack_fitness, perturbation_fitness))
