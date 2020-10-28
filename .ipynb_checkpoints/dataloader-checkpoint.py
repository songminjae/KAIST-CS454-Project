"""
load datasets, dataloader
"""

def load_dataset(dataset_name):
    if dataset_name == 'mnist':
        dataset = None
    if dataset_name == 'cifar10':
        dataset = None
    if dataset_name == 'imagenet64':
        dataset = None
    
    return dataset

def load_dataloader(dataset):
    """
    dataloader batchsize == 1
    return W,H,3 numpy arr
    """
    pass

if __name__ == '__main__':
    dataset = load_dataset('mnist')
    dataloader = load_dataloader(dataset)