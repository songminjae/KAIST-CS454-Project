import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
# how about number of noise point, nois point size? How?


# initialize perturbations with size of pop_size, with list of random gaussian distribution
def initialize_population(pop_size, origin_img_path):
    img = cv2.imread(origin_img_path)[..., ::-1]/255.0  # original image
    init_pop = []
    for i in range(pop_size):
        variance = np.random.normal(0, 0.5) + 1
        variance = np.clip(variance, 0, 2)
        noise = np.random.normal(
            0, variance, size=img.shape)  # varies variance when picking noise
        noise = noise * 0.03  # 여기 곱하는 수가 작아질 수록 원래 이미지와 비슷한 경향이 있음

        random_matrix = np.random.rand(
            img.shape[0], img.shape[1], img.shape[2])

        for i in range(len(random_matrix)):
            for j in range(len(random_matrix[i])):
                for k in range(len(random_matrix[i][j])):
                    if random_matrix[i][j][k] > 0.5:
                        noise[i][j][k] = 0.0

        init_pop.append(noise)
    # 나중에 사용할 때, np.clip(img + noise, 0, 1)으로 visible한 이미지 만들 수 있을 듯
    init_img = []
    for x in noise:
        ran = np.random.rand()
        if ran < 0.5:
            result = np.clip((img + x), 0, 1)
        else:
            result = np.clip((img*(1 + x)), 0, 1)
        init_img.append(result)

    return init_img


def fast_non_dominated_sort(fitness_list, final):
    fronts = [[]]
    S = [[] for i in range(len(fitness_list))]
    n = [-1 for i in range(len(fitness_list))]

    for p in range(len(fitness_list)):
        S[p] = []
        n[p] = 0
        for q in range(len(fitness_list))[p+1:]:
            if dominate(fitness_list[p], fitness_list[q]):
                n[q] = n[q] + 1
                S[p].append(q)
            elif dominate(fitness_list[q], fitness_list[p]):
                n[p] = n[p] + 1
                S[q].append(p)
        if n[p] == 0:
            if p not in front[0]:
                fronts[0].append(p)

    if not final:
        cnt = 0
        while (fronts[cnt]):
            next_front = []
            for p in fronts[cnt]:
                for q in S[p]:
                    n[q] = n[q] - 1
                    if n[q] == 0:
                        if q not in next_front:
                            next_front.append(q)
            fronts.append(next_front)
            cnt = cnt + 1
        fronts = fronts.pop()

    for front in fronts:
        for index in front:
            front[index] = fitness_list[front[index]]

    return fronts


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


def quick_sort(combined_list):
    if len(combined_list) <= 1: return combined_list
    pivot = len(combined_list) // 2
    left, middle, right = [], [], []
    for i in range(len(combined_list)):
        if combined_list[i][0] > combined_list[pivot][0]:
            left.append((combined_list[i]))
        elif combined_list[i][0] < combined_list[pivot][0]:
            right.append((combined_list[i]))
        else:
            middle.append((combined_list[i]))
    return quick_sort(left) + middle + quick_sort(right)


def selection(pop_size, fronts):
    N = 0
    new_pop = []
    while N < pop_size:
        for i in range(len(fronts)):
            N = N + len(fronts[i])
            if N > pop_size:
                combined_list = []
                cdlist = crowding_distance(fronts[i])
                for j in range(len(fronts[i])):
                    combined_list.append([cdlist[j], fronts[i][j])
                sorted_combined_list= quick_sort(combined_list)
                cnt= 0
                while len(new_pop) < pop_size:
                    new_pop.append(sorted_combined_list[cnt][1])
                    cnt = cnt + 1
                break
            else:
                new_pop.extend(fronts[i])

    return new_pop


def crossover(prob_c, p1, p2):  # input of 2 different perturbation with np.array
    a1, a2= np.array(p1), np.array(p2)
    assert a1.shape == a2.shape
    b= np.random.randint(0, 2, a1.shape)
    b_not= np.ones(a1.shape, int) - b
    a1_cross, a2_cross= a1 * b + a2 * b_not, a1 * b_not + a2 * b
    return a1_cross, a2_cross


def mutation(prob_m, p):
    a= np.array(p)
    c= np.ones(a.shape)
    # mutation 이 생기는 pixel은 몇개?? not mentioned in paper...?
    c[np.random.randint(a.shape[0])][np.random.randint(
        a.shape[1])]= np.random.uniform(0, 2)
    if np.random.rand() < prob_m:
        a= a * c
    return a


def do_selection_crossover_mutation(pop_size, pop, origin_img_path):
    offs= []
    s_img= selection(pop)
    # TODO: 아래 2개 함수 입력변수 지정 필요
    # c = crossover()
    # m = mutation()
    s = []
    for x in s_img:
        s.append(img_to_perturbation(x, origin_img_path))

    # pop_size는 무조건 짝수여야함!!!!
    for i in range(len(s)/2):
        a1= s[2*i]
        a2= s[2*i + 1]
        new_a1, new_a2 = crossover(0.5, a1, a2)  # HP hyperparameter prob_c
        new_a1 = mutation(0.003, new_a1)
        new_a2= mutation(0.003, new_a2)  # HP hyperparameter prob_m
        offs.append(new_a1)
        offs.append(new_a2)
    ###
    offs_img= []
    origin_img= cv2.imread(origin_img_path)[..., ::-1]/255.0  # original image

    for child in offs:
        offs_img.append(np.clip(origin_img + child), 0, 1)

    return offs_img


def fitness(pop, model, image, fitness_fn):
    fitness_fn_1, fitness_fn_2= fitness_fn
    fit_1= []
    fit_2= []
    for chromosome in enumerate(pop):
        fit_1.append(fitness_fn_1(confidence(chromosome), confidence(image)))
        fit_2.append(fitness_fn_2(chromosome, image))
    fit_list= []
    for (a, b) in (fit_1, fit_2):
        fit_list.append(a, b)
    return fit_list


def confidence(img):
    img= torch.Tensor(img).unsqueeze(0)
    img= img.cuda()
    with torch.no_grad():
        logits= model(img)
        prediction = F.softmax(logits, dim = -1)
        prediction= prediction.cpu().detach().numpy()

    return prediction[0]


def img_to_perturbation(img, origin_path):
    origin_img = cv2.imread(origin_path)[..., ::-1]/255.0
    assert(img.shape == origin_img.shape)
    return img - origin_img




def run_NSGA2(model, image, pop_size, n_generation, fitness_fn):
    """
    하나의 이미지가 들어왔을 때 NSGA2 알고리즘을 활용하여 2개의 fitness function에 대한 pareto-front를 구한다.
    위의 initialize_population, fast_non_dominated_sort, crowding_distance, crossover, mutation 등의 논문의 함수를 구현하여 사용한다.

    최종 아웃풋 형태 : pareto_front = [[f1,f2], [f1,f2], ...] : list of lists

    + MJ : image 를 origin image 의 Path로 하는게 좋을 것 같음
    """

    pareto_front= []

    iteration= 1
    parent= initialize_population(pop_size, image)
    pareto_front= fast_non_dominated_sort(fitness(parent, model, image, fitness_fn), 0)
    offspring= do_selection_crossover_mutation(pop_size, parent, image) # TODO

    while (iteration < n_generation):
        temp= parent + offspring
        pareto_front= fast_non_dominated_sort(fitness(temp, model, image, fitness_fn), 0)
        parent= []
        cnt= 0
        while (len(parent) + len(pareto_front[cnt]) < pop_size):
            parent= parent + pareto_front[cnt]
            cnt= cnt + 1
        parent= parent + pareto_front[cnt][:(pop_size - len(parent))]
        offspring= do_selection_crossover_mutation(pop_size, parent, image)
        iteration= iteration + 1

    return fast_non_dominated_sort(fitness(parent, model, image, fitness_fn), 1)


def visualize(pareto_front, path):
    """
    pareto front를 이미지 파일로 저장한다.
    """

    x= []
    y= []
    for i in range(len(pareto_front)):
        x.append(pareto_front[i][0])
        y.append(pareto_front[i][1])
    plt.title('Pareto fronts')
    plt.xlabel('Objective 1')  # name to be specified later
    plt.ylabel('Objective 2')  # name to be specified later
    plt.scatter(x, y)
    plt.savefig('fronts.png', bbox_inches='tight')

    return


if __name__ == '__main__':
    from fitness import attack_fitness, perturbation_fitness
    import cv2

    pareto_front= run_NSGA2(model, image, pop_size=10, n_generation=50, fitness_fn=(
        attack_fitness, perturbation_fitness))
