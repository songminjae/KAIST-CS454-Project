import numpy as np
from evaluation import query_cnt

def attack_fitness(query_output, target_output):
    # (list[float], list[bool]) -> float
    ####################################################
    ####################################################
    ################ implement in here #################
    ####################################################
    ####################################################

    query_cnt+=1

    y0 = target_output.argmax()
    p_y0 = query_output[y0]
    y1 = query_output.argmax()
    if (y1==y0):
        y2 = np.partition(query_output, -2)[-2]
        p_y2 = np.where(query_output == y2)
        return (p_y2 - p_y0)[0][0]

    else:
        p_y1 = y1 # np.where(query_output == y1)
        return p_y1 - p_y0


def perturbation_fitness(query_image, target_image):
    # (image, image) -> float
    # Calculate Z-Metric(equal with -Z(A^t_i))
    ####################################################
    ####################################################
    ################ implement in here #################
    ####################################################
    ####################################################
    
    #Adopted from POBA-GA paper
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