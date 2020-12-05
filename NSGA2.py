import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from fitness import z_status, calculate_Z0, attack_fitness, perturbation_fitness
from evaluation import query_cnt
# how about number of noise point, nois point size? How?


# initialize perturbations with size of pop_size, with list of random gaussian distribution
def initialize_population(pop_size, img):
    init_pop = []
    for i in range(pop_size):
        variance = np.random.normal(0, 1) + 1
        variance = np.clip(variance, 0, 2)

        noise = np.random.normal(
            0, variance, size=img.shape)  # varies variance when picking noise
        noise = noise * 0.03  # 여기 곱하는 수가 작아질 수록 원래 이미지와 비슷한 경향이 있음
        random_matrix = np.random.rand(
            img.shape[0], img.shape[1], img.shape[2])

        for i in range(len(random_matrix)):
            for j in range(len(random_matrix[i])):
                for k in range(len(random_matrix[i][j])):
                    if random_matrix[i][j][k] < 0.5:
                        noise[i][j][k] = 0.0
        init_pop.append(noise)

    init_img = []

    for x in init_pop:
        ran = np.random.rand()
        if ran < 0.5:
            result = np.clip((img + x), 0, 1)
        else:
            result = np.clip((img*(1 + x)), 0, 1)
        init_img.append(result)
    return init_img


# a malfunction may occur if any two of the noises are exactly same
def fast_non_dominated_sort(parent, fitness_list, final):
    fronts = [[]]
    fronts_image = [[]]
    S = [[] for i in range(len(fitness_list))]
    n = [0 for i in range(len(fitness_list))]

    for p in range(len(fitness_list)):
        for q in range(p+1, len(fitness_list)):
            if dominate(fitness_list[p], fitness_list[q]):
                n[q] = n[q] + 1
                S[p].append(fitness_list[q])
            elif dominate(fitness_list[q], fitness_list[p]):
                n[p] = n[p] + 1
                S[q].append(fitness_list[p])
    for p in range(len(fitness_list)):
        if n[p] == 0:
            fronts[0].append(fitness_list[p])
            fronts_image[0].append(parent[p])

    if not final:
        cnt = 0
        while (fronts[cnt]):
            next_front = []
            next_front_image = []
            for p in fronts[cnt]:
                idx_p = fitness_list.index(p)
                for q in S[idx_p]:
                    idx_q = fitness_list.index(q)
                    n[idx_q] = n[idx_q] - 1
                    if n[idx_q] == 0:
                        if fitness_list[idx_q] not in next_front:
                            next_front.append(fitness_list[idx_q])
                            next_front_image.append(parent[idx_q])
            if (len(next_front) > 0):
                fronts.append(next_front)
                fronts_image.append(next_front_image)
                cnt = cnt + 1
            else:
                break
    return fronts, fronts_image


def dominate(f1, f2):
    if (f1[0] > f2[0] and f1[1] < f2[1]) or (f1[0] >= f2[0] and f1[1] < f2[1]) or (f1[0] > f2[0] and f1[1] <= f2[1]):
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
    if len(combined_list) <= 1:
        return combined_list
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


def selection(pop_size, fronts, fronts_image):
    new_pop = []
    for i in range(len(fronts)):
        if len(new_pop) + len(fronts[i]) > pop_size:
            combined_list = []
            cdlist = crowding_distance(fronts[i])
            for j in range(len(fronts[i])):
                combined_list.append([cdlist[j], fronts_image[i][j]])
            sorted_combined_list = quick_sort(combined_list)
            cnt = 0
            while len(new_pop) < pop_size:
                new_pop.append(sorted_combined_list[cnt][1])
                cnt = cnt + 1
            break
        else:
            new_pop.extend(fronts_image[i])
    return new_pop


'''
###input:
    prob_c = [0,1]
    p1 = (32,32,3) shape, 각 value 는 [0,1]
    p2 = p1과 동일
###output:
    a1_cross = (32,32,3) shape, 각 value 는 [0,1]
    a2_cross = a1_cross 와 동일
'''


def crossover(prob_c, p1, p2):  # input of 2 different perturbation with np.array
    a1, a2 = np.array(p1), np.array(p2)
    assert a1.shape == a2.shape
    b = np.random.randint(0, 2, a1.shape)
    b_not = np.ones(a1.shape, int) - b
    a1_cross, a2_cross = a1 * b + a2 * b_not, a1 * b_not + a2 * b
    return a1_cross, a2_cross


'''
###input:
    prob_m = [0,1]
    p = (32,32,3) shaep, 각 value 는 [0,1]
###output:
    a = (32,32,3) shaep, 각 value 는 [0,1]
'''


def mutation(prob_m, p):
    a = np.array(p)
    c = np.ones(a.shape)
    # mutation 이 생기는 pixel은 몇개?? not mentioned in paper...?
    c[np.random.randint(a.shape[0])][np.random.randint(
        a.shape[1])] = np.random.uniform(0, 2)
    if np.random.rand() < prob_m:
        a = a * c
    return a


def do_selection_crossover_mutation(pop_size, fronts, fronts_image, origin_img):
    offs = []
    s_img = selection(pop_size, fronts, fronts_image)
    s = []
    for x in s_img:
        s.append(img_to_perturbation(x, origin_img))

    # pop_size는 무조건 짝수여야함!!!!
    for i in range(len(s)//2):
        a1 = s[2*i]
        a2 = s[2*i + 1]
        new_a1, new_a2 = crossover(0.5, a1, a2)  # HP hyperparameter prob_c
        new_a1 = mutation(0.003, new_a1)
        new_a2 = mutation(0.003, new_a2)  # HP hyperparameter prob_m
        offs.append(new_a1)
        offs.append(new_a2)
    ###
    offs_img = []

    for child in offs:
        offs_img.append(np.clip((origin_img + child), 0, 1))

    return offs_img


def fitness(pop, model, image, fitness_fn):
    global query_cnt
    fitness_fn_1, fitness_fn_2 = fitness_fn
    fit_1 = []
    fit_2 = []
    before_flag = z_status.Z0_flag

    preprocess = get_preprocess(size=32, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    preprocessed_image = preprocess(image)
    conf_img = confidence(model, preprocessed_image)
    for chromosome in pop:
        conf_chr = confidence(model, preprocess(chromosome))
        f1 = fitness_fn_1(conf_chr, conf_img)
        f2 = fitness_fn_2(preprocess(chromosome), preprocessed_image)
        fit_1.append(f1)
        fit_2.append(f2)
    if (before_flag == 0 and z_status.Z0_flag == 1):
        calculate_Z0(pop, preprocessed_image, preprocess)
        return fitness(pop, model, image, fitness_fn)
    query_cnt+=len(pop)
    fit_list = []
    for a, b in zip(fit_1, fit_2):
        fit_list.append([a, b])
    return fit_list


def confidence(model, img):
    img = torch.Tensor(img).unsqueeze(0)
    img = img.cuda()
    with torch.no_grad():
        logits = model(img)
        prediction = F.softmax(logits, dim=-1)
        prediction = prediction.cpu().detach().numpy()

    return prediction[0]


def img_to_perturbation(img, origin_img):
    return img - origin_img


def run_NSGA2(model, image, pop_size, n_generation, fitness_fn):
    """
    하나의 이미지가 들어왔을 때 NSGA2 알고리즘을 활용하여 2개의 fitness function에 대한 pareto-front를 구한다.
    위의 initialize_population, fast_non_dominated_sort, crowding_distance, crossover, mutation 등의 논문의 함수를 구현하여 사용한다.

    최종 아웃풋 형태 : pareto_front = [[f1,f2], [f1,f2], ...] : list of lists

    + MJ : image 를 origin image 의 Path로 하는게 좋을 것 같음
    """

    iteration = 1

    parent = initialize_population(pop_size, image)
    pareto_front, pareto_front_image = fast_non_dominated_sort(
        parent, fitness(parent, model, image, fitness_fn), 0)
    offspring = do_selection_crossover_mutation(
        pop_size, pareto_front, pareto_front_image, image)

    while (iteration < n_generation):
        temp = parent + offspring
        pareto_front, pareto_front_image = fast_non_dominated_sort(
            temp, fitness(temp, model, image, fitness_fn), 0)
        parent = []
        cnt = 0
        while (len(parent) + len(pareto_front_image[cnt]) < pop_size):
            parent = parent + pareto_front_image[cnt]
            cnt = cnt + 1
            if cnt >= len(pareto_front_image)-1:
                break
        parent = parent + pareto_front_image[cnt][:(pop_size - len(parent))]
        offspring = do_selection_crossover_mutation(
            pop_size, pareto_front, pareto_front_image, image)
        iteration = iteration + 1

    return fast_non_dominated_sort(parent, fitness(parent, model, image, fitness_fn), 1)


def visualize(pareto_front):
    """
    pareto front를 이미지 파일로 저장한다.
    """

    x = []
    y = []
    for i in range(len(pareto_front)):
        x.append(pareto_front[i][0])
        y.append(pareto_front[i][1])
    plt.title('Pareto fronts')
    plt.xlabel('Objective 1')  # name to be specified later
    plt.ylabel('Objective 2')  # name to be specified later
    plt.scatter(x, y)
    plt.savefig('fronts.png', bbox_inches='tight')

    return


""" [FOR TEST, TO BE ERASED]


def attack_fitness(query_output, target_output):
    # (list[float], list[bool]) -> float
    ####################################################
    ####################################################
    ################ implement in here #################
    ####################################################
    ####################################################
    y0 = target_output.argmax()
    p_y0 = query_output[y0]
    y1 = query_output.argmax()
    if (y1 == y0):
        y2 = np.partition(query_output, -2)[-2]
        p_y2 = np.where(query_output == y2)
        return (p_y2 - p_y0)[0][0]

    else:
        p_y1 = y1  # np.where(query_output == y1)
        return p_y1 - p_y0


def perturbation_fitness(query_image, target_image):
    # (image, image) -> float
    # Calculate Z-Metric(equal with -Z(A^t_i))
    ####################################################
    ####################################################
    ################ implement in here #################
    ####################################################
    ####################################################

    # Adopted from POBA-GA paper
    pm1 = 10
    pm2 = 5.8

    m_a, m_b, _ = target_image.shape
    pert_image = query_image - target_image
    pert_image.pow(2)
    pert_image.sum(dim=2)

    pert_image.abs()
    pert_image = -np.abs(pert_image) * pm1 + pm2
    pert_image = -1/(1+np.exp(pert_image))

    return np.sum(pert_image.numpy())


def load_dataset(dataset_name):
    def _transform(pil):
        return np.array(pil).astype('float32')/255.
    if dataset_name == 'mnist':
        dataset = MNIST(root='./datasets/', train=False,
                        transform=_transform, download=True)
    if dataset_name == 'cifar10':
        dataset = CIFAR10(root='./datasets/', train=False,
                          transform=_transform, download=True)
    if dataset_name == 'imagenet':
        dataset = ImageNet(root='./datasets/', split='val',
                           transform=_transform, download=True)
    return dataset


def load_dataloader(dataset, batch_size=1):

    dataloader batchsize == 1
    return W,H,3 numpy arr

    def _collate_fn(datas):
        xs = []
        ys = []
        for x, y in datas:
            xs.append(x)
            ys.append(y)
        return xs, ys
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=4, shuffle=False, collate_fn=_collate_fn)
    return dataloader


[FOR TEST, TO BE ERASED] """


def get_preprocess(size, mean=0, std=1):
    #mean = np.array(mean)[...,np.newaxis,np.newaxis]
    #std = np.array(std)[...,np.newaxis,np.newaxis]
    def preprocess(x):
        # W,H,3 -> 3,W,H
        x = np.rollaxis(cv2.resize(x, (size, size)), -1, 0)
        #x = (x - mean)/std
        x = torch.Tensor(x)
        return x
    return preprocess


if __name__ == '__main__':
    from models import load_model
    from torchvision import transforms
    from torchvision.datasets import MNIST, CIFAR10
    from imagenet import ImageNet
    from torch.utils.data import DataLoader
    from dataloader import *
    def give_perturbation(x):
        return x

    use_cuda = True
    dataset = load_dataset('cifar10')
    dataloader = load_dataloader(dataset)
    preprocess = get_preprocess(
        size=32, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    model = load_model('vgg16', 'cifar10', use_cuda=use_cuda)
    model.eval()

    # img = np.zeros((32,32,3)).astype('float32') ## 32,32,3 ndarray
    img = np.ones((32, 32, 3)).astype('float32')  # 32,32,3 ndarray
    img = img/2

    """
    img = give_perturbation(img)
    img = preprocess(img)
    img = torch.Tensor(img).unsqueeze(0)
    img = img.cuda()
    with torch.no_grad():
        logits = model(img)
        prediction = F.softmax(logits,dim = -1)
        prediction = prediction.cpu().detach().numpy()
    """
    pareto_front, pareto_front_image = run_NSGA2(model, img, pop_size=6, n_generation=2, fitness_fn=(
        attack_fitness, perturbation_fitness))
    visualize(pareto_front)
