import numpy as np


#Adopted from POBA-GA paper
alpha = 3
pm1 = 10
pm2 = 5.8

class Z_status():
    def __init__(self):
        self.Z0_flag = 0
        self.multiplier = 0.
    
    def reset(self):
        self.Z0_flag = 0
        self.multiplier = 0.

z_status = Z_status()

def calculate_Z0(pop, preprocessed_image, preprocess):
    tmp = 0
    for chromosome in pop:
        m_a, m_b, _ = preprocessed_image.shape
        pert_image = preprocess(chromosome) - preprocessed_image
        pert_image.pow(2)
        pert_image.sum(dim=2)

        pert_image.abs()
        pert_image = -np.abs(pert_image) * pm1 + pm2
        pert_image = (1/(1+np.exp(pert_image)) - 1/(1+np.exp(pm2)))
        
        fitness =  np.sum(pert_image.numpy())

        tmp = max(tmp, fitness)
    z_status.multiplier = -tmp * alpha


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
    if (y1==y0):
        if (z_status.Z0_flag==0):
            z_status.Z0_flag = 1
        y2 = np.partition(query_output, -2)[-2]
        p_y2 = np.where(query_output == y2)
        return (p_y2 - p_y0)[0][0]

    else:
        p_y1 = y1 # np.where(query_output == y1)
        return p_y1 - p_y0


def perturbation_fitness(query_image, target_image):
    # (image, image) -> float
    # Calculate Z-Metric(equal with -(a/Z(A0)) *Z(A^t_i))
    ####################################################
    ####################################################
    ################ implement in here #################
    ####################################################
    ####################################################

    if (z_status.Z0_flag==0):
        return 0

    m_a, m_b, _ = target_image.shape
    pert_image = query_image - target_image
    pert_image.pow(2)
    pert_image.sum(dim=2)

    pert_image.abs()
    pert_image = -np.abs(pert_image) * pm1 + pm2
    pert_image = (1/(1+np.exp(pert_image)) - 1/(1+np.exp(pm2)))
    
    return np.sum(pert_image.numpy()) * z_status.multiplier